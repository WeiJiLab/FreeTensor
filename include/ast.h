#ifndef AST_H
#define AST_H

#include <string>

#include <buffer.h>
#include <ref.h>

namespace ir {

enum class ASTNodeType : int {
    VarDef,
    Var,
    Store,
    Load,
    IntConst,
    FloatConst,
};

#define DEFINE_NODE_ACCESS(name)                                               \
  protected:                                                                   \
    name##Node() = default; /* Must be constructed in Ref */                   \
                                                                               \
    friend class Ref<name##Node>;

#define DEFINE_NODE_TRAIT(name)                                                \
    DEFINE_NODE_ACCESS(name)                                                   \
  public:                                                                      \
    virtual ASTNodeType nodeType() const override { return ASTNodeType::name; }

class ASTNode {
  public:
    virtual ~ASTNode() {}
    virtual ASTNodeType nodeType() const = 0;

    DEFINE_NODE_ACCESS(AST);
};
typedef Ref<ASTNode> AST;

class StmtNode : public ASTNode {
    DEFINE_NODE_ACCESS(Stmt);
};
typedef Ref<StmtNode> Stmt;

class ExprNode : public ASTNode {
    DEFINE_NODE_ACCESS(Expr);
};
typedef Ref<ExprNode> Expr;

class VarDefNode : public StmtNode {
  public:
    std::string name_;
    Ref<Buffer> buffer_;
    Stmt body_;

    VarDefNode(const VarDefNode &other);            // Deep copy
    VarDefNode &operator=(const VarDefNode &other); // Deep copy

    DEFINE_NODE_TRAIT(VarDef);
};
typedef Ref<VarDefNode> VarDef;
inline VarDef makeVarDef(const std::string &name, const Ref<Buffer> &buffer,
                         const Stmt &body) {
    VarDef d = VarDef::make();
    d->name_ = name, d->buffer_ = buffer, d->body_ = body;
    return d;
}

class VarNode : public ExprNode {
  public:
    std::string name_;
    DEFINE_NODE_TRAIT(Var);
};
typedef Ref<VarNode> Var;
inline Var makeVar(const std::string &name) {
    Var v = Var::make();
    v->name_ = name;
    return v;
}

class StoreNode : public StmtNode {
  public:
    Var var_;
    Expr expr_;
    std::vector<Expr> indices_;
    DEFINE_NODE_TRAIT(Store);
};
typedef Ref<StoreNode> Store;
inline Store makeStore(const Var &var, const Expr &expr,
                       const std::vector<Expr> &indices) {
    Store s = Store::make();
    s->var_ = var, s->expr_ = expr, s->indices_ = indices;
    return s;
}

class LoadNode : public ExprNode {
  public:
    Var var_;
    std::vector<Expr> indices_;
    DEFINE_NODE_TRAIT(Load);
};
typedef Ref<LoadNode> Load;
inline Load makeLoad(const Var &var, const std::vector<Expr> &indices) {
    Load l = Load::make();
    l->var_ = var, l->indices_ = indices;
    return l;
}

class IntConstNode : public ExprNode {
  public:
    int val_;
    DEFINE_NODE_TRAIT(IntConst);
};
typedef Ref<IntConstNode> IntConst;
inline IntConst makeIntConst(int val) {
    IntConst c = IntConst::make();
    c->val_ = val;
    return c;
}

class FloatConstNode : public ExprNode {
  public:
    double val_;
    DEFINE_NODE_TRAIT(FloatConst);
};
typedef Ref<FloatConstNode> FloatConst;
inline FloatConst makeFloatConst(double val) {
    FloatConst c = FloatConst::make();
    c->val_ = val;
    return c;
}

} // namespace ir

#endif // AST_H
