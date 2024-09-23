#include "moduli_autotune.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "shell_encryption/rns/rns_error_params.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr bool const debug_moduli = false;
constexpr bool const debug_graph = false;
constexpr bool const debug_output_params = true;
constexpr uint64_t const kNoiseMargin = 30;
constexpr uint64_t const kMaxPrimeBits = 58;
constexpr uint64_t const kMinPrimeBits = 12;

struct ShellParams {
  uint64_t log_n;
  uint64_t t;
  std::vector<uint64_t> qs;
};

struct ShellAutoParams {
  uint64_t cleartext_bits;
  uint64_t scaling_factor;
  uint64_t noise_offset_bits;
  uint64_t noise_variance;
  std::string seed;
};

// This function returns the uint64_t value of a scalar constant node or an
// error if this node is not a constant, scalar, or the wrong type. It is based
// on TensorFlow's GetScalarConstNodeValueHelper() in graph_utils.cc.
template <typename T, int TF_T>
Status GetScalarConstValue(NodeDef const& node, T* value) {
  if (node.op() != kConstOpName)
    return errors::InvalidArgument("Node ", node.name(),
                                   " is not a Const node. Op: ", node.op());

  Tensor tensor;
  TF_RETURN_IF_ERROR(GetNodeAttr(node, "value", &tensor));
  if (!TensorShapeUtils::IsScalar(tensor.shape())) {
    return errors::InvalidArgument(
        "Node ", node.name(),
        " should be a scalar but has shape: ", tensor.shape());
  }

  if (tensor.dtype() != TF_T) {
    return errors::InvalidArgument(
        "Node ", node.name(), " should have type ", DataTypeString(DT_UINT64),
        " but has type: ", DataTypeString(tensor.dtype()));
  }

  *value = tensor.scalar<T>()();

  return absl::OkStatus();
}

template <typename T>
Status AddScalarConstNode(T value, utils::Mutation* mutation,
                          std::string const& name, std::string const& device) {
  NodeDef node;
  node.set_op(kConstOpName);
  node.set_name(name);
  node.set_device(device);

  std::unique_ptr<tensorflow::TensorProto> tensor =
      std::make_unique<tensorflow::TensorProto>();
  std::unique_ptr<tensorflow::TensorShapeProto> tensor_shape =
      std::make_unique<tensorflow::TensorShapeProto>();

  if constexpr (std::is_same<T, uint64_t>::value) {
    (*node.mutable_attr())["dtype"].set_type(DT_UINT64);
    tensor->set_dtype(DT_UINT64);
    tensor->add_uint64_val(value);
  } else if constexpr (std::is_same<T, std::string>::value) {
    (*node.mutable_attr())["dtype"].set_type(DT_STRING);
    tensor->set_dtype(DT_STRING);
    tensor->add_string_val(value);
  } else if constexpr (std::is_same<T, std::vector<uint64_t>>::value) {
    (*node.mutable_attr())["dtype"].set_type(DT_UINT64);
    tensor->set_dtype(DT_UINT64);
    tensor_shape->add_dim()->set_size(value.size());
    for (auto const& v : value) {
      tensor->add_uint64_val(v);
    }
  } else {
    []<bool flag = false>() {
      static_assert(flag, "AddScalarConstNode does not support this type");
    }
    ();
  }
  tensor->set_allocated_tensor_shape(tensor_shape.release());
  (*node.mutable_attr())["value"].set_allocated_tensor(tensor.release());

  if constexpr (debug_graph) {
    std::cout << "Adding scalar const node: " << node.DebugString()
              << std::endl;
  }

  Status status;
  mutation->AddNode(std::move(node), &status);
  return status;
}

Status GetAutoShellContextParams(RemapperContext& ctx,
                                 ShellAutoParams& params) {
  // Get the plaintext modulus t.
  auto const* autocontext_node_view = ctx.graph_view.GetNode(kShellAutoContext);
  // auto const* autocontext_node_def = autocontext_node_view->node();

  auto const* cleartext_bits_node =
      autocontext_node_view->GetRegularFanin(0).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<uint64_t, DT_UINT64>(
      *cleartext_bits_node, &params.cleartext_bits));

  auto const* scaling_factor_node =
      autocontext_node_view->GetRegularFanin(1).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<uint64_t, DT_UINT64>(
      *scaling_factor_node, &params.scaling_factor));

  auto const* noise_offset_node =
      autocontext_node_view->GetRegularFanin(2).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<uint64_t, DT_UINT64>(
      *noise_offset_node, &params.noise_offset_bits));

  auto const* noise_variance_node =
      autocontext_node_view->GetRegularFanin(3).node_view()->node();
  TF_RETURN_IF_ERROR(GetScalarConstValue<uint64_t, DT_UINT64>(
      *noise_variance_node, &params.noise_variance));

  if constexpr (debug_moduli) {
    std::cout << "Cleartext Bits: " << params.cleartext_bits << std::endl;
    std::cout << "Scaling Factor: " << params.scaling_factor << std::endl;
    std::cout << "Noise Offset Bits: " << params.noise_offset_bits << std::endl;
    std::cout << "Noise Variance: " << params.noise_variance << std::endl;
  }

  return OkStatus();
}

int GetMulDepth(RemapperContext& ctx) {
  // Traverse the graph and return the maximum multiplicative depth.
  int const num_nodes = ctx.graph_view.NumNodes();
  std::vector<uint64_t> node_mul_depth(num_nodes);

  uint64_t max_noise = 0;
  for (int i = 0; i < num_nodes; ++i) {
    auto const* this_node_view = ctx.graph_view.GetNode(i);
    auto const* this_node_def = this_node_view->node();

    if (IsArithmetic(*this_node_def) || IsMatMul(*this_node_def) ||
        IsMulCtTfScalar(*this_node_def) || IsMulPtTfScalar(*this_node_def)) {
      // Get the fanin nodes.
      int const fanin_a_index = this_node_view->GetRegularFanin(1).node_index();
      int const fanin_b_index = this_node_view->GetRegularFanin(2).node_index();
      int max_fanin_depth = std::max(node_mul_depth[fanin_a_index],
                                     node_mul_depth[fanin_b_index]);

      // Update the multiplicative depth of this node.
      if (IsMulCtCt(*this_node_def) || IsMulCtPt(*this_node_def) ||
          IsMulPtPt(*this_node_def) || IsMatMul(*this_node_def)) {
        node_mul_depth[i] = max_fanin_depth + 1;
      } else {
        node_mul_depth[i] = max_fanin_depth;
      }
    }

    else if (IsNegCt(*this_node_def) ||
             IsFastReduceSumByRotation(*this_node_def) ||
             IsUnsortedCtSegmentSum(*this_node_def)) {
      int const fanin_a_index = this_node_view->GetRegularFanin(1).node_index();
      node_mul_depth[i] = node_mul_depth[fanin_a_index];
    }

    else if (IsRoll(*this_node_def) || IsReduceSumByRotation(*this_node_def)) {
      int const fanin_a_index = this_node_view->GetRegularFanin(2).node_index();
      node_mul_depth[i] = node_mul_depth[fanin_a_index];
    }

    else if (IsExpandDimsVariant(*this_node_def)) {
      int const fanin_a_index = this_node_view->GetRegularFanin(0).node_index();
      node_mul_depth[i] = node_mul_depth[fanin_a_index];
    } else if (IsBroadcastToShape(*this_node_def)) {
      int const fanin_a_index = this_node_view->GetRegularFanin(0).node_index();
      node_mul_depth[i] = node_mul_depth[fanin_a_index];
    } else if (IsReshape(*this_node_def)) {
      int const fanin_a_index = this_node_view->GetRegularFanin(0).node_index();
      node_mul_depth[i] = node_mul_depth[fanin_a_index];
    } else if (IsDecrypt(*this_node_def)) {
      int const fanin_a_index = this_node_view->GetRegularFanin(2).node_index();
      node_mul_depth[i] = node_mul_depth[fanin_a_index];
      max_noise = std::max(max_noise, node_mul_depth[i]);
    }
  }
  return max_noise;
}

// Function for modular exponentiation
uint64_t modPow(uint64_t base, uint64_t exp, uint64_t modulus) {
  typedef unsigned __int128 uint128_t;
  base %= modulus;
  uint128_t result = 1;
  while (exp > 0) {
    if (exp & 1) result = (result * base) % modulus;
    base = (uint128_t(base) * base) % modulus;
    exp >>= 1;
  }
  return result;
}

// Fermat's prime test.
bool CheckPrime(uint64_t const n, uint64_t const k, std::mt19937& engine) {
  if (n == 2) return true;
  if ((n & 1) == 0) return false;

  std::uniform_int_distribution<uint64_t> dist(2, n - 2);

  for (uint64_t i = 0; i < k; ++i) {
    uint64_t a = dist(engine);
    if (modPow(a, n - 1, n) != 1) return false;
  }
  return true;
}

uint64_t FindPrimeMod2n(uint64_t const two_n, uint64_t const bits_start,
                        uint64_t const bits_end,
                        std::vector<uint64_t> const not_in = {}) {
  // Prepare for random number generation.
  std::mt19937 engine(static_cast<long unsigned int>(std::time(nullptr)));

  uint64_t start = uint64_t(1) << bits_start;
  uint64_t end = uint64_t(1) << bits_end;
  for (uint64_t i = start; i < end; ++i) {
    if (CheckPrime(i, 10, engine)) {
      if (i % two_n == 1) {
        if (std::find(not_in.begin(), not_in.end(), i) == not_in.end()) {
          return i;
        }
      }
    }
  }
  return 0;
}

constexpr uint64_t BitWidth(uint64_t n) {
  uint64_t bits = 0;
  while (n) {
    n >>= 1;
    ++bits;
  }
  return bits;
}

std::vector<uint64_t> ChooseRnsCtModuli(uint64_t const two_n,
                                        uint64_t const log_t,
                                        uint64_t const log_q) {
  std::vector<uint64_t> qs;
  int64_t needed_bits_of_primes = log_q;

  // The first RNS prime is special and must be larger than the plaintext
  // modulus. This is because decryption ModReduces to the first RNS prime, then
  // to the plaintext modulus.
  uint64_t first_prime_num_bits =
      std::max(log_t + 1, static_cast<uint64_t>(needed_bits_of_primes));
  first_prime_num_bits = std::min(first_prime_num_bits, kMaxPrimeBits);
  uint64_t first_prime =
      FindPrimeMod2n(two_n, first_prime_num_bits, first_prime_num_bits + 2);
  if (first_prime == 0) {
    return {};
  }
  qs.push_back(first_prime);
  needed_bits_of_primes -= BitWidth(first_prime);

  // Find the remaining RNS primes.
  while (needed_bits_of_primes > 0) {
    uint64_t prime_num_bits =
        std::min(static_cast<uint64_t>(needed_bits_of_primes), kMaxPrimeBits);
    prime_num_bits = std::max(prime_num_bits, kMinPrimeBits);
    uint64_t prime =
        FindPrimeMod2n(two_n, prime_num_bits, prime_num_bits + 2, qs);
    if (prime == 0) {
      return {};
    }
    qs.push_back(prime);
    needed_bits_of_primes -= BitWidth(prime);
  }

  return qs;
}

uint64_t EstimateLogN(uint64_t sz) {
  // TODO: lattice-estimator.
  // return 13;
  if (sz <= 50) return 10;
  if (sz <= 70) return 11;
  if (sz <= 128) return 12;
  if (sz <= 180) return 13;
  if (sz <= 210) return 14;
  return 0;
}

Status ChooseShellParams(ShellParams& params, uint64_t const total_pt_bits,
                         uint64_t total_ct_bits) {
  // Select log_n and ciphertext moduli.
  uint64_t log_n = EstimateLogN(total_ct_bits);
  if (log_n == 0) {
    return errors::FailedPrecondition("Could not estimate log_n.");
  }
  uint64_t two_n = 1 << (log_n + 1);

  uint64_t bounded_total_pt_bits = std::max(total_pt_bits, kMinPrimeBits);

  uint64_t t =
      FindPrimeMod2n(two_n, bounded_total_pt_bits, bounded_total_pt_bits + 4);
  if (t == 0) {
    if constexpr (debug_moduli) {
      std::cout << "Could not find a prime for plaintext modulus." << std::endl;
    }
    return errors::FailedPrecondition(
        "Could not find a prime for plaintext modulus.");
  }
  int log_t = BitWidth(t);

  std::vector<uint64_t> qs = ChooseRnsCtModuli(two_n, log_t, total_ct_bits);
  if (qs.empty()) {
    if constexpr (debug_moduli) {
      std::cout << "Could not find prime(s) for ciphertext modulus."
                << std::endl;
    }
    return errors::FailedPrecondition(
        "Could not find prime(s) for ciphertext modulus.");
  }

  params.log_n = log_n;
  params.t = t;
  params.qs = std::move(qs);

  if constexpr (debug_moduli) {
    std::cout << "Choosing parameters:" << std::endl;
    std::cout << "log_n: " << params.log_n << std::endl;
    std::cout << "t: " << params.t << std::endl;
    std::cout << "qs: ";
    for (auto const& q : params.qs) {
      std::cout << q << " ";
    }
    std::cout << std::endl;
  }
  return OkStatus();
}

// Returns the noise budget of the current node.
template <typename T>
Status EstimateNodeNoise(
    RemapperContext& ctx, int node_index, std::vector<uint64_t>& node_noise,
    ShellParams const& params,
    rlwe::RnsErrorParams<rlwe::MontgomeryInt<T>>& error_params) {
  // Get the current node and its fanin nodes.
  auto const* node_view = ctx.graph_view.GetNode(node_index);
  auto const* node_def = node_view->node();

  uint64_t* this_noise = &node_noise[node_index];
  uint64_t noise_a = node_noise[node_view->GetRegularFanin(1).node_index()];
  uint64_t noise_b = node_noise[node_view->GetRegularFanin(2).node_index()];

  // Setup constants for estimating rotation noise.
  // TODO: each of these should be calculated based on the actual parameters.
  int const kNumComponents =
      2;  // TODO need to get num components based on ct*ct depth.
  constexpr int kLogGadgetBase = 4;  // TODO from rotation_kernels.cc
  int gadget_dimension = 0;
  for (auto const& q : params.qs)
    gadget_dimension += (BitWidth(q) + (kLogGadgetBase - 1)) / kLogGadgetBase;

  if (IsEncrypt(*node_def)) {
    *this_noise = BitWidth(error_params.B_secretkey_encryption());
  }

  // CtCt operations.
  else if (IsArithmetic(*node_def)) {
    if (IsAddCtCt(*node_def) || IsSubCtCt(*node_def)) {
      *this_noise = std::max(noise_a, noise_b) + 1;
    } else if (IsMulCtCt(*node_def)) {
      *this_noise = noise_a + noise_b;
    }

    // CtPt operations.
    else if (IsAddCtPt(*node_def) || IsSubCtPt(*node_def)) {
      *this_noise = noise_a;
    } else if (IsMulCtPt(*node_def)) {
      *this_noise = noise_a + BitWidth(error_params.B_plaintext());
    }
  }

  // Negation operations.
  else if (IsNegCt(*node_def)) {
    *this_noise = noise_a;
  }

  // Scalar multiplication operations.
  else if (IsMulCtTfScalar(*node_def)) {
    auto const* scalar_node_def =
        node_view->GetRegularFanin(2).node_view()->node();

    uint64_t scalar_value = 0;
    Status s = GetScalarConstValue<uint64_t, DT_UINT64>(*scalar_node_def,
                                                        &scalar_value);
    if (s.ok()) {
      *this_noise = noise_a + BitWidth(scalar_value);
    } else {
      // Try to get the scalar value from the node again as an int64.
      int64_t scalar_value = 0;
      Status s = GetScalarConstValue<int64_t, DT_INT64>(*scalar_node_def,
                                                        &scalar_value);
      int64_t abs_scalar_value = std::abs(scalar_value);
      if (!s.ok()) {
        return s;
      }
      *this_noise = noise_a + BitWidth(abs_scalar_value);
    }
  }

  // Matrix multiplication operations.
  else if (IsMatMulCtPt(*node_def)) {
    *this_noise = noise_a + BitWidth(error_params.B_plaintext());
  } else if (IsMatMulPtCt(*node_def) || IsFastMatMulPtCt(*node_def)) {
    uint64_t mul_noise = noise_a + BitWidth(error_params.B_plaintext());
    uint64_t rot_noise = BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
        kNumComponents, kLogGadgetBase, gadget_dimension));
    rot_noise += BitWidth(params.log_n);  // There are log_n rotations.
    *this_noise = std::max(mul_noise, rot_noise) + 1;
  }

  // Rotation operations.
  else if (IsRoll(*node_def)) {
    uint64_t rot_noise = BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
        kNumComponents, kLogGadgetBase, gadget_dimension));
    *this_noise = std::max(noise_b, rot_noise) + 1;
  } else if (IsReduceSumByRotation(*node_def)) {
    uint64_t rot_noise = BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
        kNumComponents, kLogGadgetBase, gadget_dimension));
    rot_noise += BitWidth(params.log_n);  // There are log_n rotations.
    *this_noise = std::max(noise_b, rot_noise);
  } else if (IsFastReduceSumByRotation(*node_def)) {
    uint64_t rot_noise = BitWidth(error_params.BoundOnGadgetBasedKeySwitching(
        kNumComponents, kLogGadgetBase, gadget_dimension));
    rot_noise += BitWidth(params.log_n);  // There are log_n rotations.
    *this_noise = std::max(noise_a, rot_noise);
  } else if (IsReduceSum(*node_def)) {
    auto const* axis_node_def =
        node_view->GetRegularFanin(2).node_view()->node();

    uint64_t axis = 0;
    TF_RETURN_IF_ERROR(
        GetScalarConstValue<uint64_t, DT_UINT64>(*axis_node_def, &axis));
    *this_noise = noise_a;  // TODO depends on axis attribute and shape.
  }

  // Segment operations.
  else if (IsUnsortedCtSegmentSum(*node_def)) {
    *this_noise = noise_a + 60;  // TODO noise is data dependent + O(slot dim).
  }

  else if (IsDecrypt(*node_def)) {
    *this_noise = noise_b;
  }

  if constexpr (debug_moduli) {
    std::cout << "\tNode " << node_def->name() << " noise bits: " << *this_noise
              << std::endl;
  }

  return OkStatus();
}

template <typename T>
Status EstimateNoiseGrowth(RemapperContext& ctx, ShellParams const& params,
                           uint64_t const noise_varaince, uint64_t* log_noise) {
  // Estimate the ciphertext noise growth by traversing the graph.
  int const num_nodes = ctx.graph_view.NumNodes();
  std::vector<uint64_t> node_noise(num_nodes);

  // Create RnsErrorParams.
  using ModularInt = rlwe::MontgomeryInt<T>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;
  std::vector<PrimeModulus*> main_moduli;
  for (auto const& q : params.qs) {
    using ModularIntParams = typename rlwe::MontgomeryInt<T>::Params;
    using NttParameters = rlwe::NttParameters<ModularInt>;

    RLWE_ASSIGN_OR_RETURN(std::unique_ptr<ModularIntParams const> mod_params_q,
                          ModularInt::Params::Create(q));
    RLWE_ASSIGN_OR_RETURN(NttParameters ntt_params_q,
                          rlwe::InitializeNttParameters<ModularInt>(
                              params.log_n, mod_params_q.get()));
    auto ntt_params_q_ptr =
        std::make_unique<NttParameters const>(std::move(ntt_params_q));
    auto modulus_q =
        new PrimeModulus{std::move(mod_params_q), std::move(ntt_params_q_ptr)};
    main_moduli.push_back(std::move(modulus_q));
  }
  auto error_params_or = rlwe::RnsErrorParams<ModularInt>::Create(
      params.log_n, main_moduli, {}, BitWidth(params.t), noise_varaince);
  if (!error_params_or.ok()) {
    return error_params_or.status();
  }
  rlwe::RnsErrorParams<ModularInt> error_params = error_params_or.value();

  uint64_t log_max_noise = 0;
  for (int i = 0; i < num_nodes; ++i) {
    // Estimate the noise budget of this node.
    TF_RETURN_IF_ERROR(
        EstimateNodeNoise<T>(ctx, i, node_noise, params, error_params));

    // Update the maximum noise budget.
    log_max_noise = std::max(log_max_noise, node_noise[i]);
  }

  if constexpr (debug_moduli) {
    std::cout << "Max noise bits: " << log_max_noise << std::endl;
  }

  *log_noise = log_max_noise;
  return OkStatus();
}

// Returns true if the node_index points to the outermost op of the pattern
// decode(encode(a)) where a is a cleartext (tf datatype) and marks nodes to
// delete accordingly.
Status ReplaceAutoparamWithContext(RemapperContext& ctx,
                                   ShellParams const& params,
                                   ShellAutoParams const& auto_params) {
  utils::MutableNodeView* autocontext_node_view =
      ctx.graph_view.GetNode(kShellAutoContext);
  int autocontext_node_index = autocontext_node_view->node_index();

  if constexpr (debug_graph) {
    std::cout << "Removing AutoShellContext node: "
              << autocontext_node_view->node()->DebugString() << std::endl;
  }

  // Create the new inputs for the ShellContext node.
  std::string log_n_name = "ContextImport64/log_n";
  std::string qs_name = "ContextImport64/main_moduli";
  std::string ps_name = "ContextImport64/aux_moduli";
  std::string t_name = "ContextImport64/plaintext_modulus";
  std::string noise_var_name = "ContextImport64/noise_variance";
  std::string seed_str_name = "ContextImport64/seed";

  utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
  std::string device = autocontext_node_view->GetDevice();
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<uint64_t>(params.log_n, mutation, log_n_name, device));
  TF_RETURN_IF_ERROR(AddScalarConstNode<std::vector<uint64_t>>(
      params.qs, mutation, qs_name, device));
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<std::vector<uint64_t>>({}, mutation, ps_name, device));
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<uint64_t>(params.t, mutation, t_name, device));
  TF_RETURN_IF_ERROR(AddScalarConstNode<uint64_t>(
      auto_params.noise_variance, mutation, noise_var_name, device));
  TF_RETURN_IF_ERROR(
      AddScalarConstNode<std::string>("", mutation, seed_str_name, device));

  // Replace the AutoShellContext node with ShellContextImport64.
  NodeDef shell_context_import_node;
  shell_context_import_node.set_op(kShellContext);
  // shell_context_import_node.set_name(autocontext_node_view->GetName());
  shell_context_import_node.set_name(kShellContext);
  shell_context_import_node.set_device(device);
  shell_context_import_node.add_input(log_n_name);
  shell_context_import_node.add_input(qs_name);
  shell_context_import_node.add_input(ps_name);
  shell_context_import_node.add_input(t_name);
  shell_context_import_node.add_input(noise_var_name);
  shell_context_import_node.add_input(seed_str_name);
  if constexpr (debug_graph) {
    std::cout << "Adding new ShellContext node: "
              << shell_context_import_node.DebugString() << std::endl;
  }
  Status status;
  mutation->AddNode(std::move(shell_context_import_node), &status);
  TF_RETURN_IF_ERROR(status);

  // Update all fanouts of the AutoShellContext node to point to the new
  // ShellContext node.
  auto const& all_fanouts = autocontext_node_view->GetRegularFanouts();
  for (int i = 0; i < static_cast<int>(all_fanouts.size()); ++i) {
    auto const& fanouts_by_port = all_fanouts[i];
    for (auto const& fanout : fanouts_by_port) {
      mutation->AddOrUpdateRegularFanin(fanout.node_view(), fanout.index(),
                                        {kShellContext, i});
      if constexpr (debug_graph) {
        std::cout << "Updating fanout: " << fanout.node_view()->node()->name()
                  << " index: " << fanout.index()
                  << " to ShellContext output port: " << i << std::endl;
      }
    }
  }

  mutation->RemoveNode(ctx.graph_view.GetNode(autocontext_node_index));
  for (auto const& fanin : autocontext_node_view->GetRegularFanins()) {
    mutation->RemoveNode(fanin.node_view());
  }

  TF_RETURN_IF_ERROR(mutation->Apply());

  return OkStatus();
}

}  // namespace

ModuliAutotuneOptimizer::ModuliAutotuneOptimizer() {}

Status ModuliAutotuneOptimizer::Init(
    tensorflow::RewriterConfig_CustomGraphOptimizer const* config) {
  return OkStatus();
}

Status ModuliAutotuneOptimizer::Optimize(Cluster* cluster,
                                         GrapplerItem const& item,
                                         GraphDef* optimized_graph) {
  GrapplerItem mutable_item = item;
  Status status;
  RemapperContext ctx(&mutable_item, &status);
  TF_RETURN_IF_ERROR(status);

  // See if an autocontext node exists in the graph. If not, there is nothing
  // to do.
  auto const* autocontext_view = ctx.graph_view.GetNode(kShellAutoContext);
  if (autocontext_view == nullptr) {
    *optimized_graph = std::move(mutable_item.graph);
    return OkStatus();
  }

  // Make sure there is only one autocontext node.
  std::string duplicate_autocontext = kShellAutoContext;
  duplicate_autocontext += "_1";
  auto const* duplicate_autocontext_view =
      ctx.graph_view.GetNode(duplicate_autocontext);
  if (duplicate_autocontext_view != nullptr) {
    return errors::FailedPrecondition("Multiple autocontext nodes found.");
  }

  // Use GetScalarConstValue to get value of plaintext modulus,
  // etc.
  ShellAutoParams auto_params;
  TF_RETURN_IF_ERROR(GetAutoShellContextParams(ctx, auto_params));

  // Topological sort so all subsequent traversals are in order.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  // Find the maximum multiplicative depth of the graph and use this to set
  // the plaintext modulus t, based on the scaling factor and depth.
  // Note the mul_depth + 1 accounts for the first multiplication by the
  // scaling factor during encoding.
  int mul_depth = GetMulDepth(ctx);
  uint64_t total_plaintext_bits =
      auto_params.cleartext_bits +
      std::ceil(std::log2(
          std::pow(auto_params.scaling_factor, std::pow(2, mul_depth + 1))));
  if constexpr (debug_moduli) {
    std::cout << "Multiplicative Depth: " << mul_depth << std::endl;
    std::cout << "Total Cleartext Bits: " << total_plaintext_bits << std::endl;
  }
  if (total_plaintext_bits > kMaxPrimeBits) {
    return errors::FailedPrecondition("Total plaintext size exceeds ",
                                      kMaxPrimeBits,
                                      ". This is not supported.");
  }

  // Do binary search to find the optimal size of the ciphertext modulus q.
  // Choose a window of moduli to search starting slightly larger than the
  // plaintext modulus. Store the best moduli in qs and ring degree in log_n.
  uint64_t log_q_l = total_plaintext_bits + 2;
  uint64_t log_q_r = total_plaintext_bits + 210;
  uint64_t log_q = (log_q_r + log_q_l) / 2;
  ShellParams params;

  while (log_q_l <= log_q_r && log_q < log_q_r + 10) {
    // The + 10 above is for the case when the binary search ends on parameters
    // which are not suitable for shell. In this case, the search will continue
    // 10 log_q's higher than the last suitable parameters found.
    Status params_status =
        ChooseShellParams(params, total_plaintext_bits, log_q);
    if (!params_status.ok()) {
      // No suitable parameters found. Try increasing log_q.
      if constexpr (debug_moduli) {
        std::cout << "No suitable parameters found for log_q: " << log_q
                  << ". Trying larger log_q." << std::endl;
      }
      log_q += 1;
      continue;
    }

    uint64_t log_max_noise = 0;
    TF_RETURN_IF_ERROR(EstimateNoiseGrowth<uint64_t>(
        ctx, params, auto_params.noise_variance, &log_max_noise));

    if (log_max_noise == 0) {
      // No encryption in this graph. Smallest parameters will do.
      log_q = log_q_l;
      break;
    }

    // Adjust the noise budget to account for the encryption noise.
    if (log_max_noise + auto_params.noise_offset_bits > log_q) {
      if constexpr (debug_moduli) {
        std::cout << "Noise budget exceeded (max noise bits: " << log_max_noise
                  << " + noise offset: " << auto_params.noise_offset_bits
                  << " > Q bits: " << log_q << ")." << std::endl;
      }
      log_q_l = log_q + 1;
    } else {
      if constexpr (debug_moduli) {
        std::cout << "Noise budget too large (max noise bits: " << log_max_noise
                  << " + noise offset: " << auto_params.noise_offset_bits
                  << " < Q bits: " << log_q << ")." << std::endl;
      }
      log_q_r = std::min(log_q_r - 1, log_q - 1);
    }
    log_q = (log_q_r + log_q_l) / 2;
  }

  if (log_q >= total_plaintext_bits + 210) {
    return errors::FailedPrecondition("Could not find suitable parameters.");
  }

  TF_RETURN_IF_ERROR(ChooseShellParams(params, total_plaintext_bits, log_q));
  if constexpr (debug_output_params) {
    std::cout << "Final parameters:" << std::endl;
    std::cout << "log_n: " << params.log_n << std::endl;
    std::cout << "t: " << params.t << std::endl;
    std::cout << "qs: ";
    for (auto const& q : params.qs) {
      std::cout << q << " ";
    }
    std::cout << std::endl;
  }

  TF_RETURN_IF_ERROR(ReplaceAutoparamWithContext(ctx, params, auto_params));

  if constexpr (debug_graph) {
    std::cout << "Optimized graph: " << std::endl;
    int const num_nodes = ctx.graph_view.NumNodes();
    for (int i = 0; i < num_nodes; ++i) {
      std::cout << ctx.graph_view.GetNode(i)->node()->DebugString()
                << std::endl;
    }
  }

  *optimized_graph = std::move(mutable_item.graph);

  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER(ModuliAutotuneOptimizer);

}  // namespace grappler
}  // namespace tensorflow