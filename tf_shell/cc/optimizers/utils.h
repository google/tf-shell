#pragma once

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"

using tensorflow::NodeDef;
using tensorflow::Status;
using tensorflow::grappler::GraphProperties;
using tensorflow::grappler::GrapplerItem;
using tensorflow::grappler::utils::MutableGraphView;
using TopMutableGraphView = tensorflow::grappler::MutableGraphView;

struct RemapperContext {
  explicit RemapperContext(GrapplerItem* item, Status* status)
      : nodes_to_preserve(item->NodesToPreserve()),
        graph_view(&item->graph, status),
        graph_properties(*item) {}

  std::unordered_set<std::string> nodes_to_preserve;
  MutableGraphView graph_view;
  GraphProperties graph_properties;
};
constexpr char kShellContext[] = "ContextImport64";
constexpr char kShellAutoContext[] = "AutoShellContext64";

constexpr char kEncode[] = "PolynomialImport64";
constexpr char kDecode[] = "PolynomialExport64";
constexpr char kEncrypt[] = "Encrypt64";
constexpr char kDecrypt[] = "Decrypt64";
constexpr char kFastDecrypt[] = "FastDecrypt64";

constexpr char kAddCtCt[] = "AddCtCt64";
constexpr char kSubCtCt[] = "SubCtCt64";
constexpr char kMulCtCt[] = "MulCtCt64";

constexpr char kAddCtPt[] = "AddCtPt64";
constexpr char kSubCtPt[] = "SubCtPt64";
constexpr char kMulCtPt[] = "MulCtPt64";

constexpr char kAddPtPt[] = "AddPtPt64";
constexpr char kSubPtPt[] = "SubPtPt64";
constexpr char kMulPtPt[] = "MulPtPt64";

constexpr char kNegPt[] = "NegPt64";
constexpr char kNegCt[] = "NegCt64";

constexpr char kMulCtTfScalar[] = "MulCtTfScalar64";
constexpr char kMulPtTfScalar[] = "MulPtTfScalar64";

constexpr char kMatMulCtPt[] = "MatMulCtPt64";
constexpr char kMatMulPtCt[] = "MatMulPtCt64";
constexpr char kFastMatMulPtCt[] = "FastMatMulPtCt64";

constexpr char kRoll[] = "Roll64";
constexpr char kReduceSumByRotation[] = "ReduceSumByRotation64";
constexpr char kFastReduceSumByRotation[] = "FastReduceSumByRotation64";
constexpr char kReduceSum[] = "ReduceSum64";

constexpr char kUnsortedCtSegmentSum[] = "UnsortedCtSegmentSum";

constexpr char kExpandDimsVariant[] = "ExpandDimsVariant";
constexpr char kBroadcastToShape[] = "BroadcastToShape";  // TODO check name
constexpr char kReshape[] = "Reshape";                    // TODO check name

constexpr char kConstOpName[] = "Const";

bool IsShellContext(NodeDef const& node);
bool IsShellAutoContext(NodeDef const& node);

bool IsEncode(NodeDef const& node);
bool IsDecode(NodeDef const& node);
bool IsEncrypt(NodeDef const& node);
bool IsPlainDerypt(NodeDef const& node);
bool IsFastDerypt(NodeDef const& node);
bool IsDecrypt(NodeDef const& node);

bool IsAddCtCt(NodeDef const& node);
bool IsSubCtCt(NodeDef const& node);
bool IsMulCtCt(NodeDef const& node);

bool IsAddCtPt(NodeDef const& node);
bool IsSubCtPt(NodeDef const& node);
bool IsMulCtPt(NodeDef const& node);

bool IsAddPtPt(NodeDef const& node);
bool IsSubPtPt(NodeDef const& node);
bool IsMulPtPt(NodeDef const& node);
bool IsArithmetic(NodeDef const& node);

bool IsNegCt(NodeDef const& node);
bool IsNegPt(NodeDef const& node);

bool IsMulCtTfScalar(NodeDef const& node);
bool IsMulPtTfScalar(NodeDef const& node);

bool IsMatMulCtPt(NodeDef const& node);
bool IsMatMulPtCt(NodeDef const& node);
bool IsFastMatMulPtCt(NodeDef const& node);
bool IsMatMul(NodeDef const& node);

bool IsRoll(NodeDef const& node);
bool IsReduceSumByRotation(NodeDef const& node);
bool IsFastReduceSumByRotation(NodeDef const& node);
bool IsReduceSum(NodeDef const& node);

bool IsUnsortedCtSegmentSum(NodeDef const& node);

bool IsExpandDimsVariant(NodeDef const& node);
bool IsBroadcastToShape(NodeDef const& node);
bool IsReshape(NodeDef const& node);