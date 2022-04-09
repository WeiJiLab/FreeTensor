#include <ast.h>
#include <hash.h>
#include <mutator.h>

namespace ir {

std::string toString(ASTNodeType type) {
    switch (type) {
#define DISPATCH(name)                                                         \
    case ASTNodeType::name:                                                    \
        return #name;

        DISPATCH(Func);
        DISPATCH(StmtSeq);
        DISPATCH(VarDef);
        DISPATCH(Store);
        DISPATCH(ReduceTo);
        DISPATCH(For);
        DISPATCH(If);
        DISPATCH(Assert);
        DISPATCH(Assume);
        DISPATCH(Eval);
        DISPATCH(Any);
        DISPATCH(Var);
        DISPATCH(Load);
        DISPATCH(IntConst);
        DISPATCH(FloatConst);
        DISPATCH(BoolConst);
        DISPATCH(Add);
        DISPATCH(Sub);
        DISPATCH(Mul);
        DISPATCH(RealDiv);
        DISPATCH(FloorDiv);
        DISPATCH(CeilDiv);
        DISPATCH(RoundTowards0Div);
        DISPATCH(Mod);
        DISPATCH(Remainder);
        DISPATCH(Min);
        DISPATCH(Max);
        DISPATCH(LT);
        DISPATCH(LE);
        DISPATCH(GT);
        DISPATCH(GE);
        DISPATCH(EQ);
        DISPATCH(NE);
        DISPATCH(LAnd);
        DISPATCH(LOr);
        DISPATCH(LNot);
        DISPATCH(Sqrt);
        DISPATCH(Exp);
        DISPATCH(Square);
        DISPATCH(Sigmoid);
        DISPATCH(Tanh);
        DISPATCH(Abs);
        DISPATCH(Floor);
        DISPATCH(Ceil);
        DISPATCH(IfExpr);
        DISPATCH(Cast);
        DISPATCH(Intrinsic);
        DISPATCH(AnyExpr);
    default:
        ERROR("Unexpected AST node type");
    }
}

AST ASTNode::parentAST() const {
    for (auto p = parent(); p.isValid(); p = p->parent()) {
        if (p->isAST()) {
            return p.as<ASTNode>();
        }
    }
    return nullptr;
}

Expr ExprNode::parentExpr() const {
    for (auto p = parentAST(); p.isValid(); p = p->parentAST()) {
        if (p->isExpr()) {
            return p.as<ExprNode>();
        }
    }
    return nullptr;
}

Stmt StmtNode::parentStmt() const {
    for (auto p = parentAST(); p.isValid(); p = p->parentAST()) {
        if (p->isStmt()) {
            return p.as<StmtNode>();
        }
    }
    return nullptr;
}

size_t ID::computeHash(const char *stmtId, Expr expr) {
    return ir::hashCombine(ir::Hasher()(expr),
                           std::hash<std::string>()(stmtId));
}

ID::ID(const Stmt &stmt) : ID(stmt->id_) {
    hash_ = computeHash(stmtId_.c_str(), expr_);
}

const std::string &ID::strId() const {
    if (expr_.isValid()) {
        ERROR("Only Stmt has strId");
    }
    return stmtId_;
}

std::string toString(const ID &id) {
    if (id.expr_.isValid()) {
        return toString(id.expr_) + " in " + id.stmtId_;
    } else {
        return id.stmtId_;
    }
}

bool operator==(const ID &lhs, const ID &rhs) {
    return lhs.stmtId_ == rhs.stmtId_ && HashComparator()(lhs.expr_, rhs.expr_);
}

bool operator!=(const ID &lhs, const ID &rhs) {
    return lhs.stmtId_ != rhs.stmtId_ ||
           !HashComparator()(lhs.expr_, rhs.expr_);
}

std::atomic<uint64_t> StmtNode::idCnt_ = 0;

std::string StmtNode::newId() { return "#" + std::to_string(idCnt_++); }

void StmtNode::setId(const ID &id) {
    if (!id.isValid()) {
        id_ = newId();
    } else {
        if (id.expr_.isValid()) {
            ERROR("Cannot assign an Expr ID to an Stmt");
        }
        id_ = id.stmtId_;
    }
}

ID StmtNode::id() const { return ID(id_); }

bool StmtNode::hasNamedId() const { return id_.empty() || id_[0] != '#'; }

Expr deepCopy(const Expr &op) { return Mutator()(op); }
Stmt deepCopy(const Stmt &op) { return Mutator()(op); }

} // namespace ir

namespace std {

size_t hash<ir::ID>::operator()(const ir::ID &id) const { return id.hash_; }

} // namespace std
