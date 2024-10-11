#include "utils.h"

#include "tensorflow/core/framework/node_def.pb.h"

bool IsShellContext(NodeDef const& node) { return node.op() == kShellContext; }
bool IsShellAutoContext(NodeDef const& node) {
  return node.op() == kShellAutoContext;
}

bool IsEncode(NodeDef const& node) { return node.op() == kEncode; }
bool IsDecode(NodeDef const& node) { return node.op() == kDecode; }
bool IsEncrypt(NodeDef const& node) { return node.op() == kEncrypt; }
bool IsPlainDerypt(NodeDef const& node) { return node.op() == kDecrypt; }
bool IsFastDecrypt(NodeDef const& node) { return node.op() == kFastDecrypt; }
bool IsDecrypt(NodeDef const& node) {
  return IsPlainDerypt(node) || IsFastDecrypt(node);
}

bool IsAddCtCt(NodeDef const& node) { return node.op() == kAddCtCt; }
bool IsSubCtCt(NodeDef const& node) { return node.op() == kSubCtCt; }
bool IsMulCtCt(NodeDef const& node) { return node.op() == kMulCtCt; }

bool IsAddCtPt(NodeDef const& node) { return node.op() == kAddCtPt; }
bool IsSubCtPt(NodeDef const& node) { return node.op() == kSubCtPt; }
bool IsMulCtPt(NodeDef const& node) { return node.op() == kMulCtPt; }

bool IsAddPtPt(NodeDef const& node) { return node.op() == kAddPtPt; }
bool IsSubPtPt(NodeDef const& node) { return node.op() == kSubPtPt; }
bool IsMulPtPt(NodeDef const& node) { return node.op() == kMulPtPt; }
bool IsArithmetic(NodeDef const& node) {
  return IsAddCtCt(node) || IsSubCtCt(node) || IsMulCtCt(node) ||
         IsAddCtPt(node) || IsSubCtPt(node) || IsMulCtPt(node) ||
         IsAddPtPt(node) || IsSubPtPt(node) || IsMulPtPt(node);
}

bool IsNegCt(NodeDef const& node) { return node.op() == kNegCt; }
bool IsNegPt(NodeDef const& node) { return node.op() == kNegPt; }

bool IsMulCtTfScalar(NodeDef const& node) {
  return node.op() == kMulCtTfScalar;
}
bool IsMulPtTfScalar(NodeDef const& node) {
  return node.op() == kMulPtTfScalar;
}

bool IsMatMulCtPt(NodeDef const& node) { return node.op() == kMatMulCtPt; }
bool IsMatMulPtCt(NodeDef const& node) { return node.op() == kMatMulPtCt; }
bool IsFastMatMulPtCt(NodeDef const& node) {
  return node.op() == kFastMatMulPtCt;
}
bool IsMatMul(NodeDef const& node) {
  return IsMatMulCtPt(node) || IsMatMulPtCt(node) || IsFastMatMulPtCt(node);
}

bool IsRoll(NodeDef const& node) { return node.op() == kRoll; }
bool IsReduceSumByRotation(NodeDef const& node) {
  return node.op() == kReduceSumByRotation;
}
bool IsFastReduceSumByRotation(NodeDef const& node) {
  return node.op() == kFastReduceSumByRotation;
}
bool IsReduceSum(NodeDef const& node) { return node.op() == kReduceSum; }

bool IsUnsortedCtSegmentSum(NodeDef const& node) {
  return node.op() == kUnsortedCtSegmentSum;
}

bool IsExpandDimsVariant(NodeDef const& node) {
  return node.op() == kExpandDimsVariant;
}

bool IsBroadcastToShape(NodeDef const& node) {
  return node.op() == kBroadcastToShape;
}

bool IsReshape(NodeDef const& node) { return node.op() == kReshape; }