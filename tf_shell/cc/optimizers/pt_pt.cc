#include "pt_pt.h"

#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr bool const debug = false;

bool IsReplaceableOp(NodeDef const& node) {
  return IsAddPtPt(node) || IsSubPtPt(node) || IsMulPtPt(node) || IsNegPt(node);
}

char const* GetReplacementOp(NodeDef const& node) {
  if (IsAddPtPt(node)) {
    return "AddV2";
  } else if (IsSubPtPt(node)) {
    return "Sub";
  } else if (IsMulPtPt(node)) {
    return "Mul";
  } else if (IsNegPt(node)) {
    return "Neg";
  }

  return nullptr;
}

struct ReorderArith {
  int shell_context_node_index;
  int encode_a_input_node;
  int encode_a_node_index;
  bool single_arg_op;
  int encode_b_input_node;
  int encode_b_node_index;
  int pt_op_node_index;
};

void PrintReorderArith(RemapperContext& ctx, ReorderArith const& reorder) {
  auto const* pt_op_node =
      ctx.graph_view.GetNode(reorder.pt_op_node_index)->node();
  auto const* encode_a_input_node =
      ctx.graph_view.GetNode(reorder.encode_a_input_node)->node();
  auto const* encode_a_node =
      ctx.graph_view.GetNode(reorder.encode_a_node_index)->node();

  std::cout << pt_op_node->name() << "( " << encode_a_node->name() << "("
            << encode_a_input_node->name() << ")";

  if (reorder.single_arg_op) {
    std::cout << " )" << std::endl;
  } else {
    auto const* encode_b_input_node =
        ctx.graph_view.GetNode(reorder.encode_b_input_node)->node();
    auto const* encode_b_node =
        ctx.graph_view.GetNode(reorder.encode_b_node_index)->node();
    std::cout << ", " << encode_b_node->name() << "("
              << encode_b_input_node->name() << ") )" << std::endl;
  }
}

// Returns true if the node_index points to the outermost op of the pattern
// outer_plaintext_op(encode(a), encode(b)) and fills the ReorderArith struct
// accordingly.
bool FindPtPt(RemapperContext& ctx, int node_index, ReorderArith* reorder) {
  auto const* outer_node_view = ctx.graph_view.GetNode(node_index);
  auto const* outer_node_def = outer_node_view->node();

  if (!IsReplaceableOp(*outer_node_def)) {
    return false;
  }

  // Check the feed nodes of the outer op are encode ops.
  auto const& outer_fanin_0 = outer_node_view->GetRegularFanin(0);
  auto const* context_node_view = outer_fanin_0.node_view();

  auto const& outer_fanin_1 = outer_node_view->GetRegularFanin(1);
  auto const* input_a_node_view = outer_fanin_1.node_view();
  auto const* input_a_node_def = input_a_node_view->node();

  // The first feed node must be an encode op.
  if (!IsEncode(*input_a_node_def)) {
    return false;
  }

  auto const& encode_a_fanin_0 = input_a_node_view->GetRegularFanin(1);
  auto const* tf_input_a_node_view = encode_a_fanin_0.node_view();

  if (IsNegPt(*outer_node_def)) {  // Negation op only has one input.
    ReorderArith new_reorder{
        .shell_context_node_index = context_node_view->node_index(),
        .encode_a_input_node = tf_input_a_node_view->node_index(),
        .encode_a_node_index = input_a_node_view->node_index(),
        .single_arg_op = true,
        .encode_b_input_node = -1,
        .encode_b_node_index = -1,
        .pt_op_node_index = node_index};

    *reorder = new_reorder;

  } else {  // All other ops have two inputs.
    auto const& outer_fanin_2 = outer_node_view->GetRegularFanin(2);
    auto const* input_b_node_view = outer_fanin_2.node_view();
    auto const* input_b_node_def = input_b_node_view->node();

    // The second feed node must also be an encode op.
    if (!IsEncode(*input_b_node_def)) {
      return false;
    }

    auto const& encode_b_fanin_0 = input_b_node_view->GetRegularFanin(1);
    auto const* tf_input_b_node_view = encode_b_fanin_0.node_view();

    ReorderArith new_reorder{
        .shell_context_node_index = context_node_view->node_index(),
        .encode_a_input_node = tf_input_a_node_view->node_index(),
        .encode_a_node_index = input_a_node_view->node_index(),
        .single_arg_op = false,
        .encode_b_input_node = tf_input_b_node_view->node_index(),
        .encode_b_node_index = input_b_node_view->node_index(),
        .pt_op_node_index = node_index};

    *reorder = new_reorder;
  }

  if constexpr (debug) {
    std::cout << "Found pattern: ";
    PrintReorderArith(ctx, *reorder);
  }

  return true;
}

// This function replaces the pattern op( encode(a), <encode(b)>) to
// encode( op(a, <b>) ) where <> indicate optional arguments. One might wonder
// why single arg ops are re-arranged in this way, since there is no performance
// gain. The reason is so subsequent ops can be optimized.
Status ApplyReorderArith(RemapperContext* ctx, ReorderArith const& reorder,
                         std::vector<bool>* nodes_to_delete) {
  GraphDef const* graph = ctx->graph_view.graph();
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;

  // First, replace the outer PtPt node with the TensorFlow equivalent and set
  // the encoder input(s) to the new node.
  NodeDef const& pt_op_node_def = graph->node(reorder.pt_op_node_index);
  auto const new_tf_op = GetReplacementOp(pt_op_node_def);
  if (new_tf_op == nullptr) {
    return errors::Internal(
        "Pt op is not supported type and cannot be replaced.");
  }

  NodeDef new_tf_op_def;
  new_tf_op_def.set_op(new_tf_op);
  std::string new_name = pt_op_node_def.name() + "_to_tf";
  new_tf_op_def.set_name(new_name.c_str());
  new_tf_op_def.set_device(pt_op_node_def.device());

  // Set the dtype of the new tf node to the same as the input.
  auto const* encode_a_node_view =
      ctx->graph_view.GetNode(reorder.encode_a_node_index);
  auto const* encode_a_node_def = encode_a_node_view->node();
  auto dtype = encode_a_node_def->attr().at("Dtype");
  new_tf_op_def.mutable_attr()->insert({"T", dtype});

  // Add the inputs to the new tf node.
  NodeDef const& input_a = graph->node(reorder.encode_a_input_node);
  new_tf_op_def.add_input(input_a.name());

  if (!reorder.single_arg_op) {
    NodeDef const& input_b = graph->node(reorder.encode_b_input_node);
    new_tf_op_def.add_input(input_b.name());
  }

  // Second, encode the output of the Tf node. Note the name of the new encode
  // node needs to be the same as the old output, even though the op is
  // different, so downstream nodes can still find it.
  NodeDef new_encode_op_def;
  new_encode_op_def.set_op(kEncode);
  new_encode_op_def.set_name(pt_op_node_def.name());  // Same as orig output.
  new_encode_op_def.set_device(pt_op_node_def.device());

  NodeDef const& shell_context_node =
      graph->node(reorder.shell_context_node_index);
  new_encode_op_def.add_input(shell_context_node.name());
  new_encode_op_def.add_input(new_tf_op_def.name());
  new_encode_op_def.mutable_attr()->insert({"Dtype", dtype});

  if constexpr (debug) {
    std::cout << "New tf_op node: \n"
              << new_tf_op_def.DebugString() << std::endl;
    std::cout << "New encode node: \n"
              << new_encode_op_def.DebugString() << std::endl;
  }

  // Add the new nodes to the graph.
  mutation->AddNode(std::move(new_tf_op_def), &status);
  TF_RETURN_IF_ERROR(status);
  mutation->AddNode(std::move(new_encode_op_def), &status);
  TF_RETURN_IF_ERROR(status);

  // Last, if the remaining encoded inputs are not used elsewhere, add them
  // to the deleted list. After the final pass through the graph, all encoder
  // nodes which do not feed encryption ops will appear in the deletion list.
  (*nodes_to_delete)[reorder.pt_op_node_index] = true;

  // auto const* encode_a_node_view =
  //     ctx->graph_view.GetNode(reorder.encode_a_node_index);

  if (encode_a_node_view->NumRegularFanouts() == 1) {
    (*nodes_to_delete)[reorder.encode_a_node_index] = true;
    if constexpr (debug) {
      std::cout << "Marked " << encode_a_node_view->node()->name()
                << " for deletion." << std::endl;
    }
  } else {
    if constexpr (debug) {
      std::cout << "Keeping " << encode_a_node_view->node()->name()
                << " with fanout " << encode_a_node_view->NumRegularFanouts()
                << std::endl;
    }
  }

  if (!reorder.single_arg_op) {
    auto const* encode_b_node_view =
        ctx->graph_view.GetNode(reorder.encode_b_node_index);

    if (encode_b_node_view->NumRegularFanouts() == 1) {
      (*nodes_to_delete)[reorder.encode_b_node_index] = true;
      if constexpr (debug) {
        std::cout << "Marked " << encode_b_node_view->node()->name()
                  << " for deletion." << std::endl;
      }
    } else {
      if constexpr (debug) {
        std::cout << "Keeping " << encode_b_node_view->node()->name()
                  << " with fanout " << encode_b_node_view->NumRegularFanouts()
                  << std::endl;
      }
    }
  }

  return OkStatus();
}

// Returns true if the node_index points to the outermost op of the pattern
// decode(encode(a)) where a is a cleartext (tf datatype) and marks nodes to
// delete accordingly.
bool FindAndRemapEncDec(RemapperContext& ctx, int node_index,
                        utils::Mutation* mutation) {
  auto const* decode_node_view = ctx.graph_view.GetNode(node_index);
  auto const* decode_node_def = decode_node_view->node();

  if (!IsDecode(*decode_node_def)) {
    return false;
  }

  // Check the feed nodes of the outer op are encode ops.
  auto const& decode_fanin_1 = decode_node_view->GetRegularFanin(1);
  auto const* encode_node_view = decode_fanin_1.node_view();
  auto const* encode_node_def = encode_node_view->node();

  // The first feed node must be an encode op.
  if (!IsEncode(*encode_node_def)) {
    return false;
  }

  auto const& encode_fanin = encode_node_view->GetRegularFanin(1);
  auto const* tf_input_node_view = encode_fanin.node_view();

  // Rename the tf input node with the decoder node name. This is necessary if
  // the decoder node is an output of the graph. Downstream nodes automatically
  // pick up on their new fanin inputs after the rename.
  utils::MutableNodeView* mutable_tf_input =
      ctx.graph_view.GetNode(tf_input_node_view->node_index());
  mutation->UpdateNodeName(mutable_tf_input, decode_node_def->name());

  // Delete the decode node.
  mutation->RemoveNode(ctx.graph_view.GetNode(node_index));

  // Only delete the encode node if it is not used elsewhere.
  if (encode_node_view->NumRegularFanouts() == 1) {
    mutation->RemoveNode(
        ctx.graph_view.GetNode(encode_node_view->node_index()));
  }

  return true;
}

}  // namespace

PtPtOptimizer::PtPtOptimizer() {}

Status PtPtOptimizer::Init(
    tensorflow::RewriterConfig_CustomGraphOptimizer const* config) {
  return OkStatus();
}

Status PtPtOptimizer::Optimize(Cluster* cluster, GrapplerItem const& item,
                               GraphDef* optimized_graph) {
  GrapplerItem mutable_item = item;
  Status status;
  RemapperContext ctx(&mutable_item, &status);
  TF_RETURN_IF_ERROR(status);

  // Topological sort and process the nodes in order.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  bool finished = false;
  while (!finished) {
    int const num_nodes = mutable_item.graph.node_size();
    std::vector<bool> nodes_to_delete(num_nodes);
    finished = true;

    for (int i = 0; i < num_nodes; ++i) {
      if (nodes_to_delete[i]) {
        continue;
      }

      // Remap op( encode(a), <encode(b)>) to encode(op(a, <b>)) where <>
      // indicate optional arguments. E.g. op=add has two arguments while
      // op=negate has only one.
      ReorderArith reorder;
      if (FindPtPt(ctx, i, &reorder)) {
        TF_RETURN_IF_ERROR(ApplyReorderArith(&ctx, reorder, &nodes_to_delete));
        finished = false;
      }
    }

    // Remove nodes.
    utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
    for (int i = 0; i < num_nodes; ++i) {
      if (nodes_to_delete[i]) {
        mutation->RemoveNode(ctx.graph_view.GetNode(i));
      }
    }
    TF_RETURN_IF_ERROR(mutation->Apply());
  }

  // Make one final backward pass to remove any remaining encode-decode pairs.
  // Since encode-decode pairs will never be nested, i.e.
  // decode(decode(encode(encode(...))), only one pass is necessary.
  {
    utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
    int const num_nodes = mutable_item.graph.node_size();
    for (int i = num_nodes - 1; i >= 0; --i) {
      FindAndRemapEncDec(ctx, i, mutation);
    }
    TF_RETURN_IF_ERROR(mutation->Apply());
  }

  *optimized_graph = std::move(mutable_item.graph);

  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER(PtPtOptimizer);

}  // namespace grappler
}  // namespace tensorflow