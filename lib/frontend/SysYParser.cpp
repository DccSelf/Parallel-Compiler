
// Generated from SysY.g4 by ANTLR 4.8


#include "frontend/SysYVisitor.h"

#include "frontend/SysYParser.h"


using namespace antlrcpp;
using namespace antlr4;

SysYParser::SysYParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

SysYParser::~SysYParser() {
  delete _interpreter;
}

std::string SysYParser::getGrammarFileName() const {
  return "SysY.g4";
}

const std::vector<std::string>& SysYParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& SysYParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- CompUnitContext ------------------------------------------------------------------

SysYParser::CompUnitContext::CompUnitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::CompUnitContext::EOF() {
  return getToken(SysYParser::EOF, 0);
}

std::vector<SysYParser::DeclContext *> SysYParser::CompUnitContext::decl() {
  return getRuleContexts<SysYParser::DeclContext>();
}

SysYParser::DeclContext* SysYParser::CompUnitContext::decl(size_t i) {
  return getRuleContext<SysYParser::DeclContext>(i);
}

std::vector<SysYParser::FuncDefContext *> SysYParser::CompUnitContext::funcDef() {
  return getRuleContexts<SysYParser::FuncDefContext>();
}

SysYParser::FuncDefContext* SysYParser::CompUnitContext::funcDef(size_t i) {
  return getRuleContext<SysYParser::FuncDefContext>(i);
}


size_t SysYParser::CompUnitContext::getRuleIndex() const {
  return SysYParser::RuleCompUnit;
}


antlrcpp::Any SysYParser::CompUnitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitCompUnit(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::CompUnitContext* SysYParser::compUnit() {
  CompUnitContext *_localctx = _tracker.createInstance<CompUnitContext>(_ctx, getState());
  enterRule(_localctx, 0, SysYParser::RuleCompUnit);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(66); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(66);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx)) {
      case 1: {
        setState(64);
        decl();
        break;
      }

      case 2: {
        setState(65);
        funcDef();
        break;
      }

      }
      setState(68); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SysYParser::Int)
      | (1ULL << SysYParser::Float)
      | (1ULL << SysYParser::Tensor)
      | (1ULL << SysYParser::Void)
      | (1ULL << SysYParser::Const))) != 0));
    setState(70);
    match(SysYParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclContext ------------------------------------------------------------------

SysYParser::DeclContext::DeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::ConstDeclContext* SysYParser::DeclContext::constDecl() {
  return getRuleContext<SysYParser::ConstDeclContext>(0);
}

SysYParser::VarDeclContext* SysYParser::DeclContext::varDecl() {
  return getRuleContext<SysYParser::VarDeclContext>(0);
}


size_t SysYParser::DeclContext::getRuleIndex() const {
  return SysYParser::RuleDecl;
}


antlrcpp::Any SysYParser::DeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitDecl(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::DeclContext* SysYParser::decl() {
  DeclContext *_localctx = _tracker.createInstance<DeclContext>(_ctx, getState());
  enterRule(_localctx, 2, SysYParser::RuleDecl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(74);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::Const: {
        enterOuterAlt(_localctx, 1);
        setState(72);
        constDecl();
        break;
      }

      case SysYParser::Int:
      case SysYParser::Float:
      case SysYParser::Tensor: {
        enterOuterAlt(_localctx, 2);
        setState(73);
        varDecl();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstDeclContext ------------------------------------------------------------------

SysYParser::ConstDeclContext::ConstDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::ConstDeclContext::Const() {
  return getToken(SysYParser::Const, 0);
}

SysYParser::BTypeContext* SysYParser::ConstDeclContext::bType() {
  return getRuleContext<SysYParser::BTypeContext>(0);
}

std::vector<SysYParser::ConstDefContext *> SysYParser::ConstDeclContext::constDef() {
  return getRuleContexts<SysYParser::ConstDefContext>();
}

SysYParser::ConstDefContext* SysYParser::ConstDeclContext::constDef(size_t i) {
  return getRuleContext<SysYParser::ConstDefContext>(i);
}

tree::TerminalNode* SysYParser::ConstDeclContext::Semicolon() {
  return getToken(SysYParser::Semicolon, 0);
}

std::vector<tree::TerminalNode *> SysYParser::ConstDeclContext::Comma() {
  return getTokens(SysYParser::Comma);
}

tree::TerminalNode* SysYParser::ConstDeclContext::Comma(size_t i) {
  return getToken(SysYParser::Comma, i);
}


size_t SysYParser::ConstDeclContext::getRuleIndex() const {
  return SysYParser::RuleConstDecl;
}


antlrcpp::Any SysYParser::ConstDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitConstDecl(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ConstDeclContext* SysYParser::constDecl() {
  ConstDeclContext *_localctx = _tracker.createInstance<ConstDeclContext>(_ctx, getState());
  enterRule(_localctx, 4, SysYParser::RuleConstDecl);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(76);
    match(SysYParser::Const);
    setState(77);
    bType();
    setState(78);
    constDef();
    setState(83);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::Comma) {
      setState(79);
      match(SysYParser::Comma);
      setState(80);
      constDef();
      setState(85);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(86);
    match(SysYParser::Semicolon);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BTypeContext ------------------------------------------------------------------

SysYParser::BTypeContext::BTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::BTypeContext::Int() {
  return getToken(SysYParser::Int, 0);
}

tree::TerminalNode* SysYParser::BTypeContext::Float() {
  return getToken(SysYParser::Float, 0);
}

tree::TerminalNode* SysYParser::BTypeContext::Tensor() {
  return getToken(SysYParser::Tensor, 0);
}


size_t SysYParser::BTypeContext::getRuleIndex() const {
  return SysYParser::RuleBType;
}


antlrcpp::Any SysYParser::BTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBType(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::BTypeContext* SysYParser::bType() {
  BTypeContext *_localctx = _tracker.createInstance<BTypeContext>(_ctx, getState());
  enterRule(_localctx, 6, SysYParser::RuleBType);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(88);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SysYParser::Int)
      | (1ULL << SysYParser::Float)
      | (1ULL << SysYParser::Tensor))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstDefContext ------------------------------------------------------------------

SysYParser::ConstDefContext::ConstDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::ConstDefContext::Identifier() {
  return getToken(SysYParser::Identifier, 0);
}

tree::TerminalNode* SysYParser::ConstDefContext::Assign() {
  return getToken(SysYParser::Assign, 0);
}

SysYParser::ConstInitValContext* SysYParser::ConstDefContext::constInitVal() {
  return getRuleContext<SysYParser::ConstInitValContext>(0);
}

std::vector<tree::TerminalNode *> SysYParser::ConstDefContext::Lbrkt() {
  return getTokens(SysYParser::Lbrkt);
}

tree::TerminalNode* SysYParser::ConstDefContext::Lbrkt(size_t i) {
  return getToken(SysYParser::Lbrkt, i);
}

std::vector<SysYParser::ConstExpContext *> SysYParser::ConstDefContext::constExp() {
  return getRuleContexts<SysYParser::ConstExpContext>();
}

SysYParser::ConstExpContext* SysYParser::ConstDefContext::constExp(size_t i) {
  return getRuleContext<SysYParser::ConstExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::ConstDefContext::Rbrkt() {
  return getTokens(SysYParser::Rbrkt);
}

tree::TerminalNode* SysYParser::ConstDefContext::Rbrkt(size_t i) {
  return getToken(SysYParser::Rbrkt, i);
}


size_t SysYParser::ConstDefContext::getRuleIndex() const {
  return SysYParser::RuleConstDef;
}


antlrcpp::Any SysYParser::ConstDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitConstDef(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ConstDefContext* SysYParser::constDef() {
  ConstDefContext *_localctx = _tracker.createInstance<ConstDefContext>(_ctx, getState());
  enterRule(_localctx, 8, SysYParser::RuleConstDef);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(90);
    match(SysYParser::Identifier);
    setState(97);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::Lbrkt) {
      setState(91);
      match(SysYParser::Lbrkt);
      setState(92);
      constExp();
      setState(93);
      match(SysYParser::Rbrkt);
      setState(99);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(100);
    match(SysYParser::Assign);
    setState(101);
    constInitVal();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstInitValContext ------------------------------------------------------------------

SysYParser::ConstInitValContext::ConstInitValContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::ConstInitValContext::getRuleIndex() const {
  return SysYParser::RuleConstInitVal;
}

void SysYParser::ConstInitValContext::copyFrom(ConstInitValContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- ListConstInitValContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::ListConstInitValContext::Lbrace() {
  return getToken(SysYParser::Lbrace, 0);
}

tree::TerminalNode* SysYParser::ListConstInitValContext::Rbrace() {
  return getToken(SysYParser::Rbrace, 0);
}

std::vector<SysYParser::ConstInitValContext *> SysYParser::ListConstInitValContext::constInitVal() {
  return getRuleContexts<SysYParser::ConstInitValContext>();
}

SysYParser::ConstInitValContext* SysYParser::ListConstInitValContext::constInitVal(size_t i) {
  return getRuleContext<SysYParser::ConstInitValContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::ListConstInitValContext::Comma() {
  return getTokens(SysYParser::Comma);
}

tree::TerminalNode* SysYParser::ListConstInitValContext::Comma(size_t i) {
  return getToken(SysYParser::Comma, i);
}

SysYParser::ListConstInitValContext::ListConstInitValContext(ConstInitValContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ListConstInitValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitListConstInitVal(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ScalarConstInitValContext ------------------------------------------------------------------

SysYParser::ConstExpContext* SysYParser::ScalarConstInitValContext::constExp() {
  return getRuleContext<SysYParser::ConstExpContext>(0);
}

SysYParser::ScalarConstInitValContext::ScalarConstInitValContext(ConstInitValContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ScalarConstInitValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitScalarConstInitVal(this);
  else
    return visitor->visitChildren(this);
}
SysYParser::ConstInitValContext* SysYParser::constInitVal() {
  ConstInitValContext *_localctx = _tracker.createInstance<ConstInitValContext>(_ctx, getState());
  enterRule(_localctx, 10, SysYParser::RuleConstInitVal);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(116);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::Lparen:
      case SysYParser::Minus:
      case SysYParser::Exclamation:
      case SysYParser::Addition:
      case SysYParser::IntConst:
      case SysYParser::FloatConst:
      case SysYParser::Identifier: {
        _localctx = dynamic_cast<ConstInitValContext *>(_tracker.createInstance<SysYParser::ScalarConstInitValContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(103);
        constExp();
        break;
      }

      case SysYParser::Lbrace: {
        _localctx = dynamic_cast<ConstInitValContext *>(_tracker.createInstance<SysYParser::ListConstInitValContext>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(104);
        match(SysYParser::Lbrace);
        setState(113);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << SysYParser::Lparen)
          | (1ULL << SysYParser::Lbrace)
          | (1ULL << SysYParser::Minus)
          | (1ULL << SysYParser::Exclamation)
          | (1ULL << SysYParser::Addition)
          | (1ULL << SysYParser::IntConst)
          | (1ULL << SysYParser::FloatConst)
          | (1ULL << SysYParser::Identifier))) != 0)) {
          setState(105);
          constInitVal();
          setState(110);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while (_la == SysYParser::Comma) {
            setState(106);
            match(SysYParser::Comma);
            setState(107);
            constInitVal();
            setState(112);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
        }
        setState(115);
        match(SysYParser::Rbrace);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDeclContext ------------------------------------------------------------------

SysYParser::VarDeclContext::VarDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::BTypeContext* SysYParser::VarDeclContext::bType() {
  return getRuleContext<SysYParser::BTypeContext>(0);
}

std::vector<SysYParser::VarDefContext *> SysYParser::VarDeclContext::varDef() {
  return getRuleContexts<SysYParser::VarDefContext>();
}

SysYParser::VarDefContext* SysYParser::VarDeclContext::varDef(size_t i) {
  return getRuleContext<SysYParser::VarDefContext>(i);
}

tree::TerminalNode* SysYParser::VarDeclContext::Semicolon() {
  return getToken(SysYParser::Semicolon, 0);
}

std::vector<tree::TerminalNode *> SysYParser::VarDeclContext::Comma() {
  return getTokens(SysYParser::Comma);
}

tree::TerminalNode* SysYParser::VarDeclContext::Comma(size_t i) {
  return getToken(SysYParser::Comma, i);
}


size_t SysYParser::VarDeclContext::getRuleIndex() const {
  return SysYParser::RuleVarDecl;
}


antlrcpp::Any SysYParser::VarDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitVarDecl(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::VarDeclContext* SysYParser::varDecl() {
  VarDeclContext *_localctx = _tracker.createInstance<VarDeclContext>(_ctx, getState());
  enterRule(_localctx, 12, SysYParser::RuleVarDecl);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(118);
    bType();
    setState(119);
    varDef();
    setState(124);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::Comma) {
      setState(120);
      match(SysYParser::Comma);
      setState(121);
      varDef();
      setState(126);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(127);
    match(SysYParser::Semicolon);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDefContext ------------------------------------------------------------------

SysYParser::VarDefContext::VarDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::VarDefContext::getRuleIndex() const {
  return SysYParser::RuleVarDef;
}

void SysYParser::VarDefContext::copyFrom(VarDefContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- UninitVarDefContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::UninitVarDefContext::Identifier() {
  return getToken(SysYParser::Identifier, 0);
}

std::vector<tree::TerminalNode *> SysYParser::UninitVarDefContext::Lbrkt() {
  return getTokens(SysYParser::Lbrkt);
}

tree::TerminalNode* SysYParser::UninitVarDefContext::Lbrkt(size_t i) {
  return getToken(SysYParser::Lbrkt, i);
}

std::vector<SysYParser::ConstExpContext *> SysYParser::UninitVarDefContext::constExp() {
  return getRuleContexts<SysYParser::ConstExpContext>();
}

SysYParser::ConstExpContext* SysYParser::UninitVarDefContext::constExp(size_t i) {
  return getRuleContext<SysYParser::ConstExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::UninitVarDefContext::Rbrkt() {
  return getTokens(SysYParser::Rbrkt);
}

tree::TerminalNode* SysYParser::UninitVarDefContext::Rbrkt(size_t i) {
  return getToken(SysYParser::Rbrkt, i);
}

SysYParser::UninitVarDefContext::UninitVarDefContext(VarDefContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::UninitVarDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitUninitVarDef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- InitVarDefContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::InitVarDefContext::Identifier() {
  return getToken(SysYParser::Identifier, 0);
}

tree::TerminalNode* SysYParser::InitVarDefContext::Assign() {
  return getToken(SysYParser::Assign, 0);
}

SysYParser::InitValContext* SysYParser::InitVarDefContext::initVal() {
  return getRuleContext<SysYParser::InitValContext>(0);
}

std::vector<tree::TerminalNode *> SysYParser::InitVarDefContext::Lbrkt() {
  return getTokens(SysYParser::Lbrkt);
}

tree::TerminalNode* SysYParser::InitVarDefContext::Lbrkt(size_t i) {
  return getToken(SysYParser::Lbrkt, i);
}

std::vector<SysYParser::ConstExpContext *> SysYParser::InitVarDefContext::constExp() {
  return getRuleContexts<SysYParser::ConstExpContext>();
}

SysYParser::ConstExpContext* SysYParser::InitVarDefContext::constExp(size_t i) {
  return getRuleContext<SysYParser::ConstExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::InitVarDefContext::Rbrkt() {
  return getTokens(SysYParser::Rbrkt);
}

tree::TerminalNode* SysYParser::InitVarDefContext::Rbrkt(size_t i) {
  return getToken(SysYParser::Rbrkt, i);
}

SysYParser::InitVarDefContext::InitVarDefContext(VarDefContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::InitVarDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitInitVarDef(this);
  else
    return visitor->visitChildren(this);
}
SysYParser::VarDefContext* SysYParser::varDef() {
  VarDefContext *_localctx = _tracker.createInstance<VarDefContext>(_ctx, getState());
  enterRule(_localctx, 14, SysYParser::RuleVarDef);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(151);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<VarDefContext *>(_tracker.createInstance<SysYParser::UninitVarDefContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(129);
      match(SysYParser::Identifier);
      setState(136);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == SysYParser::Lbrkt) {
        setState(130);
        match(SysYParser::Lbrkt);
        setState(131);
        constExp();
        setState(132);
        match(SysYParser::Rbrkt);
        setState(138);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      break;
    }

    case 2: {
      _localctx = dynamic_cast<VarDefContext *>(_tracker.createInstance<SysYParser::InitVarDefContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(139);
      match(SysYParser::Identifier);
      setState(146);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == SysYParser::Lbrkt) {
        setState(140);
        match(SysYParser::Lbrkt);
        setState(141);
        constExp();
        setState(142);
        match(SysYParser::Rbrkt);
        setState(148);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(149);
      match(SysYParser::Assign);
      setState(150);
      initVal();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InitValContext ------------------------------------------------------------------

SysYParser::InitValContext::InitValContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::InitValContext::getRuleIndex() const {
  return SysYParser::RuleInitVal;
}

void SysYParser::InitValContext::copyFrom(InitValContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- ScalarInitValContext ------------------------------------------------------------------

SysYParser::ExpContext* SysYParser::ScalarInitValContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

SysYParser::ScalarInitValContext::ScalarInitValContext(InitValContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ScalarInitValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitScalarInitVal(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ListInitValContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::ListInitValContext::Lbrace() {
  return getToken(SysYParser::Lbrace, 0);
}

tree::TerminalNode* SysYParser::ListInitValContext::Rbrace() {
  return getToken(SysYParser::Rbrace, 0);
}

std::vector<SysYParser::InitValContext *> SysYParser::ListInitValContext::initVal() {
  return getRuleContexts<SysYParser::InitValContext>();
}

SysYParser::InitValContext* SysYParser::ListInitValContext::initVal(size_t i) {
  return getRuleContext<SysYParser::InitValContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::ListInitValContext::Comma() {
  return getTokens(SysYParser::Comma);
}

tree::TerminalNode* SysYParser::ListInitValContext::Comma(size_t i) {
  return getToken(SysYParser::Comma, i);
}

SysYParser::ListInitValContext::ListInitValContext(InitValContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ListInitValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitListInitVal(this);
  else
    return visitor->visitChildren(this);
}
SysYParser::InitValContext* SysYParser::initVal() {
  InitValContext *_localctx = _tracker.createInstance<InitValContext>(_ctx, getState());
  enterRule(_localctx, 16, SysYParser::RuleInitVal);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(166);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::Lparen:
      case SysYParser::Minus:
      case SysYParser::Exclamation:
      case SysYParser::Addition:
      case SysYParser::IntConst:
      case SysYParser::FloatConst:
      case SysYParser::Identifier: {
        _localctx = dynamic_cast<InitValContext *>(_tracker.createInstance<SysYParser::ScalarInitValContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(153);
        exp();
        break;
      }

      case SysYParser::Lbrace: {
        _localctx = dynamic_cast<InitValContext *>(_tracker.createInstance<SysYParser::ListInitValContext>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(154);
        match(SysYParser::Lbrace);
        setState(163);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << SysYParser::Lparen)
          | (1ULL << SysYParser::Lbrace)
          | (1ULL << SysYParser::Minus)
          | (1ULL << SysYParser::Exclamation)
          | (1ULL << SysYParser::Addition)
          | (1ULL << SysYParser::IntConst)
          | (1ULL << SysYParser::FloatConst)
          | (1ULL << SysYParser::Identifier))) != 0)) {
          setState(155);
          initVal();
          setState(160);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while (_la == SysYParser::Comma) {
            setState(156);
            match(SysYParser::Comma);
            setState(157);
            initVal();
            setState(162);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
        }
        setState(165);
        match(SysYParser::Rbrace);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncDefContext ------------------------------------------------------------------

SysYParser::FuncDefContext::FuncDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::FuncTypeContext* SysYParser::FuncDefContext::funcType() {
  return getRuleContext<SysYParser::FuncTypeContext>(0);
}

tree::TerminalNode* SysYParser::FuncDefContext::Identifier() {
  return getToken(SysYParser::Identifier, 0);
}

tree::TerminalNode* SysYParser::FuncDefContext::Lparen() {
  return getToken(SysYParser::Lparen, 0);
}

tree::TerminalNode* SysYParser::FuncDefContext::Rparen() {
  return getToken(SysYParser::Rparen, 0);
}

SysYParser::BlockContext* SysYParser::FuncDefContext::block() {
  return getRuleContext<SysYParser::BlockContext>(0);
}

SysYParser::FuncFParamsContext* SysYParser::FuncDefContext::funcFParams() {
  return getRuleContext<SysYParser::FuncFParamsContext>(0);
}


size_t SysYParser::FuncDefContext::getRuleIndex() const {
  return SysYParser::RuleFuncDef;
}


antlrcpp::Any SysYParser::FuncDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncDef(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncDefContext* SysYParser::funcDef() {
  FuncDefContext *_localctx = _tracker.createInstance<FuncDefContext>(_ctx, getState());
  enterRule(_localctx, 18, SysYParser::RuleFuncDef);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(168);
    funcType();
    setState(169);
    match(SysYParser::Identifier);
    setState(170);
    match(SysYParser::Lparen);
    setState(172);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SysYParser::Int)
      | (1ULL << SysYParser::Float)
      | (1ULL << SysYParser::Tensor))) != 0)) {
      setState(171);
      funcFParams();
    }
    setState(174);
    match(SysYParser::Rparen);
    setState(175);
    block();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncTypeContext ------------------------------------------------------------------

SysYParser::FuncTypeContext::FuncTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::FuncTypeContext::Void() {
  return getToken(SysYParser::Void, 0);
}

tree::TerminalNode* SysYParser::FuncTypeContext::Int() {
  return getToken(SysYParser::Int, 0);
}

tree::TerminalNode* SysYParser::FuncTypeContext::Float() {
  return getToken(SysYParser::Float, 0);
}

tree::TerminalNode* SysYParser::FuncTypeContext::Tensor() {
  return getToken(SysYParser::Tensor, 0);
}


size_t SysYParser::FuncTypeContext::getRuleIndex() const {
  return SysYParser::RuleFuncType;
}


antlrcpp::Any SysYParser::FuncTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncType(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncTypeContext* SysYParser::funcType() {
  FuncTypeContext *_localctx = _tracker.createInstance<FuncTypeContext>(_ctx, getState());
  enterRule(_localctx, 20, SysYParser::RuleFuncType);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(177);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SysYParser::Int)
      | (1ULL << SysYParser::Float)
      | (1ULL << SysYParser::Tensor)
      | (1ULL << SysYParser::Void))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncFParamsContext ------------------------------------------------------------------

SysYParser::FuncFParamsContext::FuncFParamsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SysYParser::FuncFParamContext *> SysYParser::FuncFParamsContext::funcFParam() {
  return getRuleContexts<SysYParser::FuncFParamContext>();
}

SysYParser::FuncFParamContext* SysYParser::FuncFParamsContext::funcFParam(size_t i) {
  return getRuleContext<SysYParser::FuncFParamContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::FuncFParamsContext::Comma() {
  return getTokens(SysYParser::Comma);
}

tree::TerminalNode* SysYParser::FuncFParamsContext::Comma(size_t i) {
  return getToken(SysYParser::Comma, i);
}


size_t SysYParser::FuncFParamsContext::getRuleIndex() const {
  return SysYParser::RuleFuncFParams;
}


antlrcpp::Any SysYParser::FuncFParamsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncFParams(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncFParamsContext* SysYParser::funcFParams() {
  FuncFParamsContext *_localctx = _tracker.createInstance<FuncFParamsContext>(_ctx, getState());
  enterRule(_localctx, 22, SysYParser::RuleFuncFParams);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(179);
    funcFParam();
    setState(184);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::Comma) {
      setState(180);
      match(SysYParser::Comma);
      setState(181);
      funcFParam();
      setState(186);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncFParamContext ------------------------------------------------------------------

SysYParser::FuncFParamContext::FuncFParamContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::BTypeContext* SysYParser::FuncFParamContext::bType() {
  return getRuleContext<SysYParser::BTypeContext>(0);
}

tree::TerminalNode* SysYParser::FuncFParamContext::Identifier() {
  return getToken(SysYParser::Identifier, 0);
}

std::vector<tree::TerminalNode *> SysYParser::FuncFParamContext::Lbrkt() {
  return getTokens(SysYParser::Lbrkt);
}

tree::TerminalNode* SysYParser::FuncFParamContext::Lbrkt(size_t i) {
  return getToken(SysYParser::Lbrkt, i);
}

std::vector<tree::TerminalNode *> SysYParser::FuncFParamContext::Rbrkt() {
  return getTokens(SysYParser::Rbrkt);
}

tree::TerminalNode* SysYParser::FuncFParamContext::Rbrkt(size_t i) {
  return getToken(SysYParser::Rbrkt, i);
}

std::vector<SysYParser::ConstExpContext *> SysYParser::FuncFParamContext::constExp() {
  return getRuleContexts<SysYParser::ConstExpContext>();
}

SysYParser::ConstExpContext* SysYParser::FuncFParamContext::constExp(size_t i) {
  return getRuleContext<SysYParser::ConstExpContext>(i);
}


size_t SysYParser::FuncFParamContext::getRuleIndex() const {
  return SysYParser::RuleFuncFParam;
}


antlrcpp::Any SysYParser::FuncFParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncFParam(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncFParamContext* SysYParser::funcFParam() {
  FuncFParamContext *_localctx = _tracker.createInstance<FuncFParamContext>(_ctx, getState());
  enterRule(_localctx, 24, SysYParser::RuleFuncFParam);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(187);
    bType();
    setState(188);
    match(SysYParser::Identifier);
    setState(200);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SysYParser::Lbrkt) {
      setState(189);
      match(SysYParser::Lbrkt);
      setState(190);
      match(SysYParser::Rbrkt);
      setState(197);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == SysYParser::Lbrkt) {
        setState(191);
        match(SysYParser::Lbrkt);
        setState(192);
        constExp();
        setState(193);
        match(SysYParser::Rbrkt);
        setState(199);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockContext ------------------------------------------------------------------

SysYParser::BlockContext::BlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::BlockContext::Lbrace() {
  return getToken(SysYParser::Lbrace, 0);
}

tree::TerminalNode* SysYParser::BlockContext::Rbrace() {
  return getToken(SysYParser::Rbrace, 0);
}

std::vector<SysYParser::BlockItemContext *> SysYParser::BlockContext::blockItem() {
  return getRuleContexts<SysYParser::BlockItemContext>();
}

SysYParser::BlockItemContext* SysYParser::BlockContext::blockItem(size_t i) {
  return getRuleContext<SysYParser::BlockItemContext>(i);
}


size_t SysYParser::BlockContext::getRuleIndex() const {
  return SysYParser::RuleBlock;
}


antlrcpp::Any SysYParser::BlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBlock(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::BlockContext* SysYParser::block() {
  BlockContext *_localctx = _tracker.createInstance<BlockContext>(_ctx, getState());
  enterRule(_localctx, 26, SysYParser::RuleBlock);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(202);
    match(SysYParser::Lbrace);
    setState(206);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SysYParser::Int)
      | (1ULL << SysYParser::Float)
      | (1ULL << SysYParser::Tensor)
      | (1ULL << SysYParser::Const)
      | (1ULL << SysYParser::Return)
      | (1ULL << SysYParser::If)
      | (1ULL << SysYParser::While)
      | (1ULL << SysYParser::Break)
      | (1ULL << SysYParser::Continue)
      | (1ULL << SysYParser::Lparen)
      | (1ULL << SysYParser::Lbrace)
      | (1ULL << SysYParser::Semicolon)
      | (1ULL << SysYParser::Minus)
      | (1ULL << SysYParser::Exclamation)
      | (1ULL << SysYParser::Addition)
      | (1ULL << SysYParser::IntConst)
      | (1ULL << SysYParser::FloatConst)
      | (1ULL << SysYParser::Identifier))) != 0)) {
      setState(203);
      blockItem();
      setState(208);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(209);
    match(SysYParser::Rbrace);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockItemContext ------------------------------------------------------------------

SysYParser::BlockItemContext::BlockItemContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::DeclContext* SysYParser::BlockItemContext::decl() {
  return getRuleContext<SysYParser::DeclContext>(0);
}

SysYParser::StmtContext* SysYParser::BlockItemContext::stmt() {
  return getRuleContext<SysYParser::StmtContext>(0);
}


size_t SysYParser::BlockItemContext::getRuleIndex() const {
  return SysYParser::RuleBlockItem;
}


antlrcpp::Any SysYParser::BlockItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBlockItem(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::BlockItemContext* SysYParser::blockItem() {
  BlockItemContext *_localctx = _tracker.createInstance<BlockItemContext>(_ctx, getState());
  enterRule(_localctx, 28, SysYParser::RuleBlockItem);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(213);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::Int:
      case SysYParser::Float:
      case SysYParser::Tensor:
      case SysYParser::Const: {
        enterOuterAlt(_localctx, 1);
        setState(211);
        decl();
        break;
      }

      case SysYParser::Return:
      case SysYParser::If:
      case SysYParser::While:
      case SysYParser::Break:
      case SysYParser::Continue:
      case SysYParser::Lparen:
      case SysYParser::Lbrace:
      case SysYParser::Semicolon:
      case SysYParser::Minus:
      case SysYParser::Exclamation:
      case SysYParser::Addition:
      case SysYParser::IntConst:
      case SysYParser::FloatConst:
      case SysYParser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(212);
        stmt();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StmtContext ------------------------------------------------------------------

SysYParser::StmtContext::StmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::StmtContext::getRuleIndex() const {
  return SysYParser::RuleStmt;
}

void SysYParser::StmtContext::copyFrom(StmtContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- WhileStmtContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::WhileStmtContext::While() {
  return getToken(SysYParser::While, 0);
}

tree::TerminalNode* SysYParser::WhileStmtContext::Lparen() {
  return getToken(SysYParser::Lparen, 0);
}

SysYParser::CondContext* SysYParser::WhileStmtContext::cond() {
  return getRuleContext<SysYParser::CondContext>(0);
}

tree::TerminalNode* SysYParser::WhileStmtContext::Rparen() {
  return getToken(SysYParser::Rparen, 0);
}

SysYParser::StmtContext* SysYParser::WhileStmtContext::stmt() {
  return getRuleContext<SysYParser::StmtContext>(0);
}

SysYParser::WhileStmtContext::WhileStmtContext(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::WhileStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitWhileStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BlockStmtContext ------------------------------------------------------------------

SysYParser::BlockContext* SysYParser::BlockStmtContext::block() {
  return getRuleContext<SysYParser::BlockContext>(0);
}

SysYParser::BlockStmtContext::BlockStmtContext(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::BlockStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBlockStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AssignmentContext ------------------------------------------------------------------

SysYParser::LValContext* SysYParser::AssignmentContext::lVal() {
  return getRuleContext<SysYParser::LValContext>(0);
}

tree::TerminalNode* SysYParser::AssignmentContext::Assign() {
  return getToken(SysYParser::Assign, 0);
}

SysYParser::ExpContext* SysYParser::AssignmentContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

tree::TerminalNode* SysYParser::AssignmentContext::Semicolon() {
  return getToken(SysYParser::Semicolon, 0);
}

SysYParser::AssignmentContext::AssignmentContext(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::AssignmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitAssignment(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IfStmt1Context ------------------------------------------------------------------

tree::TerminalNode* SysYParser::IfStmt1Context::If() {
  return getToken(SysYParser::If, 0);
}

tree::TerminalNode* SysYParser::IfStmt1Context::Lparen() {
  return getToken(SysYParser::Lparen, 0);
}

SysYParser::CondContext* SysYParser::IfStmt1Context::cond() {
  return getRuleContext<SysYParser::CondContext>(0);
}

tree::TerminalNode* SysYParser::IfStmt1Context::Rparen() {
  return getToken(SysYParser::Rparen, 0);
}

SysYParser::StmtContext* SysYParser::IfStmt1Context::stmt() {
  return getRuleContext<SysYParser::StmtContext>(0);
}

SysYParser::IfStmt1Context::IfStmt1Context(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::IfStmt1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitIfStmt1(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BreakStmtContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::BreakStmtContext::Break() {
  return getToken(SysYParser::Break, 0);
}

tree::TerminalNode* SysYParser::BreakStmtContext::Semicolon() {
  return getToken(SysYParser::Semicolon, 0);
}

SysYParser::BreakStmtContext::BreakStmtContext(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::BreakStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBreakStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ExpStmtContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::ExpStmtContext::Semicolon() {
  return getToken(SysYParser::Semicolon, 0);
}

SysYParser::ExpContext* SysYParser::ExpStmtContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

SysYParser::ExpStmtContext::ExpStmtContext(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ExpStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitExpStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IfStmt2Context ------------------------------------------------------------------

tree::TerminalNode* SysYParser::IfStmt2Context::If() {
  return getToken(SysYParser::If, 0);
}

tree::TerminalNode* SysYParser::IfStmt2Context::Lparen() {
  return getToken(SysYParser::Lparen, 0);
}

SysYParser::CondContext* SysYParser::IfStmt2Context::cond() {
  return getRuleContext<SysYParser::CondContext>(0);
}

tree::TerminalNode* SysYParser::IfStmt2Context::Rparen() {
  return getToken(SysYParser::Rparen, 0);
}

std::vector<SysYParser::StmtContext *> SysYParser::IfStmt2Context::stmt() {
  return getRuleContexts<SysYParser::StmtContext>();
}

SysYParser::StmtContext* SysYParser::IfStmt2Context::stmt(size_t i) {
  return getRuleContext<SysYParser::StmtContext>(i);
}

tree::TerminalNode* SysYParser::IfStmt2Context::Else() {
  return getToken(SysYParser::Else, 0);
}

SysYParser::IfStmt2Context::IfStmt2Context(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::IfStmt2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitIfStmt2(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ReturnStmtContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::ReturnStmtContext::Return() {
  return getToken(SysYParser::Return, 0);
}

tree::TerminalNode* SysYParser::ReturnStmtContext::Semicolon() {
  return getToken(SysYParser::Semicolon, 0);
}

SysYParser::ExpContext* SysYParser::ReturnStmtContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

SysYParser::ReturnStmtContext::ReturnStmtContext(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ReturnStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitReturnStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ContinueStmtContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::ContinueStmtContext::Continue() {
  return getToken(SysYParser::Continue, 0);
}

tree::TerminalNode* SysYParser::ContinueStmtContext::Semicolon() {
  return getToken(SysYParser::Semicolon, 0);
}

SysYParser::ContinueStmtContext::ContinueStmtContext(StmtContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ContinueStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitContinueStmt(this);
  else
    return visitor->visitChildren(this);
}
SysYParser::StmtContext* SysYParser::stmt() {
  StmtContext *_localctx = _tracker.createInstance<StmtContext>(_ctx, getState());
  enterRule(_localctx, 30, SysYParser::RuleStmt);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(254);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::AssignmentContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(215);
      lVal();
      setState(216);
      match(SysYParser::Assign);
      setState(217);
      exp();
      setState(218);
      match(SysYParser::Semicolon);
      break;
    }

    case 2: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::ExpStmtContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(221);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << SysYParser::Lparen)
        | (1ULL << SysYParser::Minus)
        | (1ULL << SysYParser::Exclamation)
        | (1ULL << SysYParser::Addition)
        | (1ULL << SysYParser::IntConst)
        | (1ULL << SysYParser::FloatConst)
        | (1ULL << SysYParser::Identifier))) != 0)) {
        setState(220);
        exp();
      }
      setState(223);
      match(SysYParser::Semicolon);
      break;
    }

    case 3: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::BlockStmtContext>(_localctx));
      enterOuterAlt(_localctx, 3);
      setState(224);
      block();
      break;
    }

    case 4: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::IfStmt1Context>(_localctx));
      enterOuterAlt(_localctx, 4);
      setState(225);
      match(SysYParser::If);
      setState(226);
      match(SysYParser::Lparen);
      setState(227);
      cond();
      setState(228);
      match(SysYParser::Rparen);
      setState(229);
      stmt();
      break;
    }

    case 5: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::IfStmt2Context>(_localctx));
      enterOuterAlt(_localctx, 5);
      setState(231);
      match(SysYParser::If);
      setState(232);
      match(SysYParser::Lparen);
      setState(233);
      cond();
      setState(234);
      match(SysYParser::Rparen);
      setState(235);
      stmt();
      setState(236);
      match(SysYParser::Else);
      setState(237);
      stmt();
      break;
    }

    case 6: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::WhileStmtContext>(_localctx));
      enterOuterAlt(_localctx, 6);
      setState(239);
      match(SysYParser::While);
      setState(240);
      match(SysYParser::Lparen);
      setState(241);
      cond();
      setState(242);
      match(SysYParser::Rparen);
      setState(243);
      stmt();
      break;
    }

    case 7: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::BreakStmtContext>(_localctx));
      enterOuterAlt(_localctx, 7);
      setState(245);
      match(SysYParser::Break);
      setState(246);
      match(SysYParser::Semicolon);
      break;
    }

    case 8: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::ContinueStmtContext>(_localctx));
      enterOuterAlt(_localctx, 8);
      setState(247);
      match(SysYParser::Continue);
      setState(248);
      match(SysYParser::Semicolon);
      break;
    }

    case 9: {
      _localctx = dynamic_cast<StmtContext *>(_tracker.createInstance<SysYParser::ReturnStmtContext>(_localctx));
      enterOuterAlt(_localctx, 9);
      setState(249);
      match(SysYParser::Return);
      setState(251);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << SysYParser::Lparen)
        | (1ULL << SysYParser::Minus)
        | (1ULL << SysYParser::Exclamation)
        | (1ULL << SysYParser::Addition)
        | (1ULL << SysYParser::IntConst)
        | (1ULL << SysYParser::FloatConst)
        | (1ULL << SysYParser::Identifier))) != 0)) {
        setState(250);
        exp();
      }
      setState(253);
      match(SysYParser::Semicolon);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpContext ------------------------------------------------------------------

SysYParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::AddExpContext* SysYParser::ExpContext::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}


size_t SysYParser::ExpContext::getRuleIndex() const {
  return SysYParser::RuleExp;
}


antlrcpp::Any SysYParser::ExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitExp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ExpContext* SysYParser::exp() {
  ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, getState());
  enterRule(_localctx, 32, SysYParser::RuleExp);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(256);
    addExp(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CondContext ------------------------------------------------------------------

SysYParser::CondContext::CondContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::LOrExpContext* SysYParser::CondContext::lOrExp() {
  return getRuleContext<SysYParser::LOrExpContext>(0);
}


size_t SysYParser::CondContext::getRuleIndex() const {
  return SysYParser::RuleCond;
}


antlrcpp::Any SysYParser::CondContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitCond(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::CondContext* SysYParser::cond() {
  CondContext *_localctx = _tracker.createInstance<CondContext>(_ctx, getState());
  enterRule(_localctx, 34, SysYParser::RuleCond);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(258);
    lOrExp(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LValContext ------------------------------------------------------------------

SysYParser::LValContext::LValContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::LValContext::Identifier() {
  return getToken(SysYParser::Identifier, 0);
}

std::vector<tree::TerminalNode *> SysYParser::LValContext::Lbrkt() {
  return getTokens(SysYParser::Lbrkt);
}

tree::TerminalNode* SysYParser::LValContext::Lbrkt(size_t i) {
  return getToken(SysYParser::Lbrkt, i);
}

std::vector<SysYParser::ExpContext *> SysYParser::LValContext::exp() {
  return getRuleContexts<SysYParser::ExpContext>();
}

SysYParser::ExpContext* SysYParser::LValContext::exp(size_t i) {
  return getRuleContext<SysYParser::ExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::LValContext::Rbrkt() {
  return getTokens(SysYParser::Rbrkt);
}

tree::TerminalNode* SysYParser::LValContext::Rbrkt(size_t i) {
  return getToken(SysYParser::Rbrkt, i);
}


size_t SysYParser::LValContext::getRuleIndex() const {
  return SysYParser::RuleLVal;
}


antlrcpp::Any SysYParser::LValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLVal(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::LValContext* SysYParser::lVal() {
  LValContext *_localctx = _tracker.createInstance<LValContext>(_ctx, getState());
  enterRule(_localctx, 36, SysYParser::RuleLVal);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(260);
    match(SysYParser::Identifier);
    setState(267);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(261);
        match(SysYParser::Lbrkt);
        setState(262);
        exp();
        setState(263);
        match(SysYParser::Rbrkt); 
      }
      setState(269);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrimaryExpContext ------------------------------------------------------------------

SysYParser::PrimaryExpContext::PrimaryExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::PrimaryExpContext::getRuleIndex() const {
  return SysYParser::RulePrimaryExp;
}

void SysYParser::PrimaryExpContext::copyFrom(PrimaryExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- PrimaryExp2Context ------------------------------------------------------------------

SysYParser::LValContext* SysYParser::PrimaryExp2Context::lVal() {
  return getRuleContext<SysYParser::LValContext>(0);
}

SysYParser::PrimaryExp2Context::PrimaryExp2Context(PrimaryExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::PrimaryExp2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitPrimaryExp2(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PrimaryExp1Context ------------------------------------------------------------------

tree::TerminalNode* SysYParser::PrimaryExp1Context::Lparen() {
  return getToken(SysYParser::Lparen, 0);
}

SysYParser::ExpContext* SysYParser::PrimaryExp1Context::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

tree::TerminalNode* SysYParser::PrimaryExp1Context::Rparen() {
  return getToken(SysYParser::Rparen, 0);
}

SysYParser::PrimaryExp1Context::PrimaryExp1Context(PrimaryExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::PrimaryExp1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitPrimaryExp1(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PrimaryExp3Context ------------------------------------------------------------------

SysYParser::NumberContext* SysYParser::PrimaryExp3Context::number() {
  return getRuleContext<SysYParser::NumberContext>(0);
}

SysYParser::PrimaryExp3Context::PrimaryExp3Context(PrimaryExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::PrimaryExp3Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitPrimaryExp3(this);
  else
    return visitor->visitChildren(this);
}
SysYParser::PrimaryExpContext* SysYParser::primaryExp() {
  PrimaryExpContext *_localctx = _tracker.createInstance<PrimaryExpContext>(_ctx, getState());
  enterRule(_localctx, 38, SysYParser::RulePrimaryExp);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(276);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::Lparen: {
        _localctx = dynamic_cast<PrimaryExpContext *>(_tracker.createInstance<SysYParser::PrimaryExp1Context>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(270);
        match(SysYParser::Lparen);
        setState(271);
        exp();
        setState(272);
        match(SysYParser::Rparen);
        break;
      }

      case SysYParser::Identifier: {
        _localctx = dynamic_cast<PrimaryExpContext *>(_tracker.createInstance<SysYParser::PrimaryExp2Context>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(274);
        lVal();
        break;
      }

      case SysYParser::IntConst:
      case SysYParser::FloatConst: {
        _localctx = dynamic_cast<PrimaryExpContext *>(_tracker.createInstance<SysYParser::PrimaryExp3Context>(_localctx));
        enterOuterAlt(_localctx, 3);
        setState(275);
        number();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NumberContext ------------------------------------------------------------------

SysYParser::NumberContext::NumberContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::NumberContext::IntConst() {
  return getToken(SysYParser::IntConst, 0);
}

tree::TerminalNode* SysYParser::NumberContext::FloatConst() {
  return getToken(SysYParser::FloatConst, 0);
}


size_t SysYParser::NumberContext::getRuleIndex() const {
  return SysYParser::RuleNumber;
}


antlrcpp::Any SysYParser::NumberContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitNumber(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::NumberContext* SysYParser::number() {
  NumberContext *_localctx = _tracker.createInstance<NumberContext>(_ctx, getState());
  enterRule(_localctx, 40, SysYParser::RuleNumber);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(278);
    _la = _input->LA(1);
    if (!(_la == SysYParser::IntConst

    || _la == SysYParser::FloatConst)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryExpContext ------------------------------------------------------------------

SysYParser::UnaryExpContext::UnaryExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::UnaryExpContext::getRuleIndex() const {
  return SysYParser::RuleUnaryExp;
}

void SysYParser::UnaryExpContext::copyFrom(UnaryExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Unary1Context ------------------------------------------------------------------

SysYParser::PrimaryExpContext* SysYParser::Unary1Context::primaryExp() {
  return getRuleContext<SysYParser::PrimaryExpContext>(0);
}

SysYParser::Unary1Context::Unary1Context(UnaryExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Unary1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitUnary1(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Unary2Context ------------------------------------------------------------------

tree::TerminalNode* SysYParser::Unary2Context::Identifier() {
  return getToken(SysYParser::Identifier, 0);
}

tree::TerminalNode* SysYParser::Unary2Context::Lparen() {
  return getToken(SysYParser::Lparen, 0);
}

tree::TerminalNode* SysYParser::Unary2Context::Rparen() {
  return getToken(SysYParser::Rparen, 0);
}

SysYParser::FuncRParamsContext* SysYParser::Unary2Context::funcRParams() {
  return getRuleContext<SysYParser::FuncRParamsContext>(0);
}

SysYParser::Unary2Context::Unary2Context(UnaryExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Unary2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitUnary2(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Unary3Context ------------------------------------------------------------------

SysYParser::UnaryOpContext* SysYParser::Unary3Context::unaryOp() {
  return getRuleContext<SysYParser::UnaryOpContext>(0);
}

SysYParser::UnaryExpContext* SysYParser::Unary3Context::unaryExp() {
  return getRuleContext<SysYParser::UnaryExpContext>(0);
}

SysYParser::Unary3Context::Unary3Context(UnaryExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Unary3Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitUnary3(this);
  else
    return visitor->visitChildren(this);
}
SysYParser::UnaryExpContext* SysYParser::unaryExp() {
  UnaryExpContext *_localctx = _tracker.createInstance<UnaryExpContext>(_ctx, getState());
  enterRule(_localctx, 42, SysYParser::RuleUnaryExp);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(290);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 27, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<UnaryExpContext *>(_tracker.createInstance<SysYParser::Unary1Context>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(280);
      primaryExp();
      break;
    }

    case 2: {
      _localctx = dynamic_cast<UnaryExpContext *>(_tracker.createInstance<SysYParser::Unary2Context>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(281);
      match(SysYParser::Identifier);
      setState(282);
      match(SysYParser::Lparen);
      setState(284);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << SysYParser::Lparen)
        | (1ULL << SysYParser::Minus)
        | (1ULL << SysYParser::Exclamation)
        | (1ULL << SysYParser::Addition)
        | (1ULL << SysYParser::IntConst)
        | (1ULL << SysYParser::FloatConst)
        | (1ULL << SysYParser::Identifier)
        | (1ULL << SysYParser::STRING))) != 0)) {
        setState(283);
        funcRParams();
      }
      setState(286);
      match(SysYParser::Rparen);
      break;
    }

    case 3: {
      _localctx = dynamic_cast<UnaryExpContext *>(_tracker.createInstance<SysYParser::Unary3Context>(_localctx));
      enterOuterAlt(_localctx, 3);
      setState(287);
      unaryOp();
      setState(288);
      unaryExp();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryOpContext ------------------------------------------------------------------

SysYParser::UnaryOpContext::UnaryOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::UnaryOpContext::Addition() {
  return getToken(SysYParser::Addition, 0);
}

tree::TerminalNode* SysYParser::UnaryOpContext::Minus() {
  return getToken(SysYParser::Minus, 0);
}

tree::TerminalNode* SysYParser::UnaryOpContext::Exclamation() {
  return getToken(SysYParser::Exclamation, 0);
}


size_t SysYParser::UnaryOpContext::getRuleIndex() const {
  return SysYParser::RuleUnaryOp;
}


antlrcpp::Any SysYParser::UnaryOpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitUnaryOp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::UnaryOpContext* SysYParser::unaryOp() {
  UnaryOpContext *_localctx = _tracker.createInstance<UnaryOpContext>(_ctx, getState());
  enterRule(_localctx, 44, SysYParser::RuleUnaryOp);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(292);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << SysYParser::Minus)
      | (1ULL << SysYParser::Exclamation)
      | (1ULL << SysYParser::Addition))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncRParamsContext ------------------------------------------------------------------

SysYParser::FuncRParamsContext::FuncRParamsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SysYParser::FuncRParamContext *> SysYParser::FuncRParamsContext::funcRParam() {
  return getRuleContexts<SysYParser::FuncRParamContext>();
}

SysYParser::FuncRParamContext* SysYParser::FuncRParamsContext::funcRParam(size_t i) {
  return getRuleContext<SysYParser::FuncRParamContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::FuncRParamsContext::Comma() {
  return getTokens(SysYParser::Comma);
}

tree::TerminalNode* SysYParser::FuncRParamsContext::Comma(size_t i) {
  return getToken(SysYParser::Comma, i);
}


size_t SysYParser::FuncRParamsContext::getRuleIndex() const {
  return SysYParser::RuleFuncRParams;
}


antlrcpp::Any SysYParser::FuncRParamsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncRParams(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncRParamsContext* SysYParser::funcRParams() {
  FuncRParamsContext *_localctx = _tracker.createInstance<FuncRParamsContext>(_ctx, getState());
  enterRule(_localctx, 46, SysYParser::RuleFuncRParams);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(294);
    funcRParam();
    setState(299);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::Comma) {
      setState(295);
      match(SysYParser::Comma);
      setState(296);
      funcRParam();
      setState(301);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncRParamContext ------------------------------------------------------------------

SysYParser::FuncRParamContext::FuncRParamContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::FuncRParamContext::getRuleIndex() const {
  return SysYParser::RuleFuncRParam;
}

void SysYParser::FuncRParamContext::copyFrom(FuncRParamContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- StringAsRParamContext ------------------------------------------------------------------

tree::TerminalNode* SysYParser::StringAsRParamContext::STRING() {
  return getToken(SysYParser::STRING, 0);
}

SysYParser::StringAsRParamContext::StringAsRParamContext(FuncRParamContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::StringAsRParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitStringAsRParam(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ExpAsRParamContext ------------------------------------------------------------------

SysYParser::ExpContext* SysYParser::ExpAsRParamContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

SysYParser::ExpAsRParamContext::ExpAsRParamContext(FuncRParamContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::ExpAsRParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitExpAsRParam(this);
  else
    return visitor->visitChildren(this);
}
SysYParser::FuncRParamContext* SysYParser::funcRParam() {
  FuncRParamContext *_localctx = _tracker.createInstance<FuncRParamContext>(_ctx, getState());
  enterRule(_localctx, 48, SysYParser::RuleFuncRParam);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(304);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::Lparen:
      case SysYParser::Minus:
      case SysYParser::Exclamation:
      case SysYParser::Addition:
      case SysYParser::IntConst:
      case SysYParser::FloatConst:
      case SysYParser::Identifier: {
        _localctx = dynamic_cast<FuncRParamContext *>(_tracker.createInstance<SysYParser::ExpAsRParamContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(302);
        exp();
        break;
      }

      case SysYParser::STRING: {
        _localctx = dynamic_cast<FuncRParamContext *>(_tracker.createInstance<SysYParser::StringAsRParamContext>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(303);
        match(SysYParser::STRING);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MulExpContext ------------------------------------------------------------------

SysYParser::MulExpContext::MulExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::MulExpContext::getRuleIndex() const {
  return SysYParser::RuleMulExp;
}

void SysYParser::MulExpContext::copyFrom(MulExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Mul2Context ------------------------------------------------------------------

SysYParser::MulExpContext* SysYParser::Mul2Context::mulExp() {
  return getRuleContext<SysYParser::MulExpContext>(0);
}

SysYParser::UnaryExpContext* SysYParser::Mul2Context::unaryExp() {
  return getRuleContext<SysYParser::UnaryExpContext>(0);
}

tree::TerminalNode* SysYParser::Mul2Context::Multiplication() {
  return getToken(SysYParser::Multiplication, 0);
}

tree::TerminalNode* SysYParser::Mul2Context::Division() {
  return getToken(SysYParser::Division, 0);
}

tree::TerminalNode* SysYParser::Mul2Context::Modulo() {
  return getToken(SysYParser::Modulo, 0);
}

SysYParser::Mul2Context::Mul2Context(MulExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Mul2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitMul2(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Mul1Context ------------------------------------------------------------------

SysYParser::UnaryExpContext* SysYParser::Mul1Context::unaryExp() {
  return getRuleContext<SysYParser::UnaryExpContext>(0);
}

SysYParser::Mul1Context::Mul1Context(MulExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Mul1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitMul1(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::MulExpContext* SysYParser::mulExp() {
   return mulExp(0);
}

SysYParser::MulExpContext* SysYParser::mulExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::MulExpContext *_localctx = _tracker.createInstance<MulExpContext>(_ctx, parentState);
  SysYParser::MulExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 50;
  enterRecursionRule(_localctx, 50, SysYParser::RuleMulExp, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    _localctx = _tracker.createInstance<Mul1Context>(_localctx);
    _ctx = _localctx;
    previousContext = _localctx;

    setState(307);
    unaryExp();
    _ctx->stop = _input->LT(-1);
    setState(314);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<Mul2Context>(_tracker.createInstance<MulExpContext>(parentContext, parentState));
        _localctx = newContext;
        pushNewRecursionContext(newContext, startState, RuleMulExp);
        setState(309);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(310);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << SysYParser::Multiplication)
          | (1ULL << SysYParser::Division)
          | (1ULL << SysYParser::Modulo))) != 0))) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(311);
        unaryExp(); 
      }
      setState(316);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- AddExpContext ------------------------------------------------------------------

SysYParser::AddExpContext::AddExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::AddExpContext::getRuleIndex() const {
  return SysYParser::RuleAddExp;
}

void SysYParser::AddExpContext::copyFrom(AddExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Add2Context ------------------------------------------------------------------

SysYParser::AddExpContext* SysYParser::Add2Context::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}

SysYParser::MulExpContext* SysYParser::Add2Context::mulExp() {
  return getRuleContext<SysYParser::MulExpContext>(0);
}

tree::TerminalNode* SysYParser::Add2Context::Addition() {
  return getToken(SysYParser::Addition, 0);
}

tree::TerminalNode* SysYParser::Add2Context::Minus() {
  return getToken(SysYParser::Minus, 0);
}

SysYParser::Add2Context::Add2Context(AddExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Add2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitAdd2(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Add1Context ------------------------------------------------------------------

SysYParser::MulExpContext* SysYParser::Add1Context::mulExp() {
  return getRuleContext<SysYParser::MulExpContext>(0);
}

SysYParser::Add1Context::Add1Context(AddExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Add1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitAdd1(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::AddExpContext* SysYParser::addExp() {
   return addExp(0);
}

SysYParser::AddExpContext* SysYParser::addExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::AddExpContext *_localctx = _tracker.createInstance<AddExpContext>(_ctx, parentState);
  SysYParser::AddExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 52;
  enterRecursionRule(_localctx, 52, SysYParser::RuleAddExp, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    _localctx = _tracker.createInstance<Add1Context>(_localctx);
    _ctx = _localctx;
    previousContext = _localctx;

    setState(318);
    mulExp(0);
    _ctx->stop = _input->LT(-1);
    setState(325);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<Add2Context>(_tracker.createInstance<AddExpContext>(parentContext, parentState));
        _localctx = newContext;
        pushNewRecursionContext(newContext, startState, RuleAddExp);
        setState(320);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(321);
        _la = _input->LA(1);
        if (!(_la == SysYParser::Minus

        || _la == SysYParser::Addition)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(322);
        mulExp(0); 
      }
      setState(327);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- RelExpContext ------------------------------------------------------------------

SysYParser::RelExpContext::RelExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::RelExpContext::getRuleIndex() const {
  return SysYParser::RuleRelExp;
}

void SysYParser::RelExpContext::copyFrom(RelExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Rel2Context ------------------------------------------------------------------

SysYParser::RelExpContext* SysYParser::Rel2Context::relExp() {
  return getRuleContext<SysYParser::RelExpContext>(0);
}

SysYParser::AddExpContext* SysYParser::Rel2Context::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}

tree::TerminalNode* SysYParser::Rel2Context::LT() {
  return getToken(SysYParser::LT, 0);
}

tree::TerminalNode* SysYParser::Rel2Context::GT() {
  return getToken(SysYParser::GT, 0);
}

tree::TerminalNode* SysYParser::Rel2Context::LE() {
  return getToken(SysYParser::LE, 0);
}

tree::TerminalNode* SysYParser::Rel2Context::GE() {
  return getToken(SysYParser::GE, 0);
}

SysYParser::Rel2Context::Rel2Context(RelExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Rel2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitRel2(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Rel1Context ------------------------------------------------------------------

SysYParser::AddExpContext* SysYParser::Rel1Context::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}

SysYParser::Rel1Context::Rel1Context(RelExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Rel1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitRel1(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::RelExpContext* SysYParser::relExp() {
   return relExp(0);
}

SysYParser::RelExpContext* SysYParser::relExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::RelExpContext *_localctx = _tracker.createInstance<RelExpContext>(_ctx, parentState);
  SysYParser::RelExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 54;
  enterRecursionRule(_localctx, 54, SysYParser::RuleRelExp, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    _localctx = _tracker.createInstance<Rel1Context>(_localctx);
    _ctx = _localctx;
    previousContext = _localctx;

    setState(329);
    addExp(0);
    _ctx->stop = _input->LT(-1);
    setState(336);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 32, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<Rel2Context>(_tracker.createInstance<RelExpContext>(parentContext, parentState));
        _localctx = newContext;
        pushNewRecursionContext(newContext, startState, RuleRelExp);
        setState(331);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(332);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << SysYParser::LT)
          | (1ULL << SysYParser::LE)
          | (1ULL << SysYParser::GT)
          | (1ULL << SysYParser::GE))) != 0))) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(333);
        addExp(0); 
      }
      setState(338);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 32, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- EqExpContext ------------------------------------------------------------------

SysYParser::EqExpContext::EqExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::EqExpContext::getRuleIndex() const {
  return SysYParser::RuleEqExp;
}

void SysYParser::EqExpContext::copyFrom(EqExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Eq1Context ------------------------------------------------------------------

SysYParser::RelExpContext* SysYParser::Eq1Context::relExp() {
  return getRuleContext<SysYParser::RelExpContext>(0);
}

SysYParser::Eq1Context::Eq1Context(EqExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Eq1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitEq1(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Eq2Context ------------------------------------------------------------------

SysYParser::EqExpContext* SysYParser::Eq2Context::eqExp() {
  return getRuleContext<SysYParser::EqExpContext>(0);
}

SysYParser::RelExpContext* SysYParser::Eq2Context::relExp() {
  return getRuleContext<SysYParser::RelExpContext>(0);
}

tree::TerminalNode* SysYParser::Eq2Context::EQ() {
  return getToken(SysYParser::EQ, 0);
}

tree::TerminalNode* SysYParser::Eq2Context::NEQ() {
  return getToken(SysYParser::NEQ, 0);
}

SysYParser::Eq2Context::Eq2Context(EqExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::Eq2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitEq2(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::EqExpContext* SysYParser::eqExp() {
   return eqExp(0);
}

SysYParser::EqExpContext* SysYParser::eqExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::EqExpContext *_localctx = _tracker.createInstance<EqExpContext>(_ctx, parentState);
  SysYParser::EqExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 56;
  enterRecursionRule(_localctx, 56, SysYParser::RuleEqExp, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    _localctx = _tracker.createInstance<Eq1Context>(_localctx);
    _ctx = _localctx;
    previousContext = _localctx;

    setState(340);
    relExp(0);
    _ctx->stop = _input->LT(-1);
    setState(347);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<Eq2Context>(_tracker.createInstance<EqExpContext>(parentContext, parentState));
        _localctx = newContext;
        pushNewRecursionContext(newContext, startState, RuleEqExp);
        setState(342);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(343);
        _la = _input->LA(1);
        if (!(_la == SysYParser::EQ

        || _la == SysYParser::NEQ)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(344);
        relExp(0); 
      }
      setState(349);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- LAndExpContext ------------------------------------------------------------------

SysYParser::LAndExpContext::LAndExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::LAndExpContext::getRuleIndex() const {
  return SysYParser::RuleLAndExp;
}

void SysYParser::LAndExpContext::copyFrom(LAndExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- LAnd2Context ------------------------------------------------------------------

SysYParser::LAndExpContext* SysYParser::LAnd2Context::lAndExp() {
  return getRuleContext<SysYParser::LAndExpContext>(0);
}

tree::TerminalNode* SysYParser::LAnd2Context::LAND() {
  return getToken(SysYParser::LAND, 0);
}

SysYParser::EqExpContext* SysYParser::LAnd2Context::eqExp() {
  return getRuleContext<SysYParser::EqExpContext>(0);
}

SysYParser::LAnd2Context::LAnd2Context(LAndExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::LAnd2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLAnd2(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LAnd1Context ------------------------------------------------------------------

SysYParser::EqExpContext* SysYParser::LAnd1Context::eqExp() {
  return getRuleContext<SysYParser::EqExpContext>(0);
}

SysYParser::LAnd1Context::LAnd1Context(LAndExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::LAnd1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLAnd1(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::LAndExpContext* SysYParser::lAndExp() {
   return lAndExp(0);
}

SysYParser::LAndExpContext* SysYParser::lAndExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::LAndExpContext *_localctx = _tracker.createInstance<LAndExpContext>(_ctx, parentState);
  SysYParser::LAndExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 58;
  enterRecursionRule(_localctx, 58, SysYParser::RuleLAndExp, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    _localctx = _tracker.createInstance<LAnd1Context>(_localctx);
    _ctx = _localctx;
    previousContext = _localctx;

    setState(351);
    eqExp(0);
    _ctx->stop = _input->LT(-1);
    setState(358);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<LAnd2Context>(_tracker.createInstance<LAndExpContext>(parentContext, parentState));
        _localctx = newContext;
        pushNewRecursionContext(newContext, startState, RuleLAndExp);
        setState(353);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(354);
        match(SysYParser::LAND);
        setState(355);
        eqExp(0); 
      }
      setState(360);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- LOrExpContext ------------------------------------------------------------------

SysYParser::LOrExpContext::LOrExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t SysYParser::LOrExpContext::getRuleIndex() const {
  return SysYParser::RuleLOrExp;
}

void SysYParser::LOrExpContext::copyFrom(LOrExpContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- LOr1Context ------------------------------------------------------------------

SysYParser::LAndExpContext* SysYParser::LOr1Context::lAndExp() {
  return getRuleContext<SysYParser::LAndExpContext>(0);
}

SysYParser::LOr1Context::LOr1Context(LOrExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::LOr1Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLOr1(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LOr2Context ------------------------------------------------------------------

SysYParser::LOrExpContext* SysYParser::LOr2Context::lOrExp() {
  return getRuleContext<SysYParser::LOrExpContext>(0);
}

tree::TerminalNode* SysYParser::LOr2Context::LOR() {
  return getToken(SysYParser::LOR, 0);
}

SysYParser::LAndExpContext* SysYParser::LOr2Context::lAndExp() {
  return getRuleContext<SysYParser::LAndExpContext>(0);
}

SysYParser::LOr2Context::LOr2Context(LOrExpContext *ctx) { copyFrom(ctx); }


antlrcpp::Any SysYParser::LOr2Context::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLOr2(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::LOrExpContext* SysYParser::lOrExp() {
   return lOrExp(0);
}

SysYParser::LOrExpContext* SysYParser::lOrExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::LOrExpContext *_localctx = _tracker.createInstance<LOrExpContext>(_ctx, parentState);
  SysYParser::LOrExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 60;
  enterRecursionRule(_localctx, 60, SysYParser::RuleLOrExp, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    _localctx = _tracker.createInstance<LOr1Context>(_localctx);
    _ctx = _localctx;
    previousContext = _localctx;

    setState(362);
    lAndExp(0);
    _ctx->stop = _input->LT(-1);
    setState(369);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 35, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<LOr2Context>(_tracker.createInstance<LOrExpContext>(parentContext, parentState));
        _localctx = newContext;
        pushNewRecursionContext(newContext, startState, RuleLOrExp);
        setState(364);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(365);
        match(SysYParser::LOR);
        setState(366);
        lAndExp(0); 
      }
      setState(371);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 35, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ConstExpContext ------------------------------------------------------------------

SysYParser::ConstExpContext::ConstExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::AddExpContext* SysYParser::ConstExpContext::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}


size_t SysYParser::ConstExpContext::getRuleIndex() const {
  return SysYParser::RuleConstExp;
}


antlrcpp::Any SysYParser::ConstExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitConstExp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ConstExpContext* SysYParser::constExp() {
  ConstExpContext *_localctx = _tracker.createInstance<ConstExpContext>(_ctx, getState());
  enterRule(_localctx, 62, SysYParser::RuleConstExp);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(372);
    addExp(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool SysYParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 25: return mulExpSempred(dynamic_cast<MulExpContext *>(context), predicateIndex);
    case 26: return addExpSempred(dynamic_cast<AddExpContext *>(context), predicateIndex);
    case 27: return relExpSempred(dynamic_cast<RelExpContext *>(context), predicateIndex);
    case 28: return eqExpSempred(dynamic_cast<EqExpContext *>(context), predicateIndex);
    case 29: return lAndExpSempred(dynamic_cast<LAndExpContext *>(context), predicateIndex);
    case 30: return lOrExpSempred(dynamic_cast<LOrExpContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool SysYParser::mulExpSempred(MulExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::addExpSempred(AddExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 1: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::relExpSempred(RelExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 2: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::eqExpSempred(EqExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 3: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::lAndExpSempred(LAndExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 4: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::lOrExpSempred(LOrExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 5: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> SysYParser::_decisionToDFA;
atn::PredictionContextCache SysYParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN SysYParser::_atn;
std::vector<uint16_t> SysYParser::_serializedATN;

std::vector<std::string> SysYParser::_ruleNames = {
  "compUnit", "decl", "constDecl", "bType", "constDef", "constInitVal", 
  "varDecl", "varDef", "initVal", "funcDef", "funcType", "funcFParams", 
  "funcFParam", "block", "blockItem", "stmt", "exp", "cond", "lVal", "primaryExp", 
  "number", "unaryExp", "unaryOp", "funcRParams", "funcRParam", "mulExp", 
  "addExp", "relExp", "eqExp", "lAndExp", "lOrExp", "constExp"
};

std::vector<std::string> SysYParser::_literalNames = {
  "", "'int'", "'float'", "'tensor'", "'void'", "'const'", "'return'", "'if'", 
  "'else'", "'for'", "'while'", "'break'", "'continue'", "'('", "')'", "'['", 
  "']'", "'{'", "'}'", "','", "';'", "'?'", "':'", "'='", "'-'", "'!'", 
  "'~'", "'+'", "'*'", "'/'", "'%'", "'&&'", "'||'", "'=='", "'!='", "'<'", 
  "'<='", "'>'", "'>='"
};

std::vector<std::string> SysYParser::_symbolicNames = {
  "", "Int", "Float", "Tensor", "Void", "Const", "Return", "If", "Else", 
  "For", "While", "Break", "Continue", "Lparen", "Rparen", "Lbrkt", "Rbrkt", 
  "Lbrace", "Rbrace", "Comma", "Semicolon", "Question", "Colon", "Assign", 
  "Minus", "Exclamation", "Tilde", "Addition", "Multiplication", "Division", 
  "Modulo", "LAND", "LOR", "EQ", "NEQ", "LT", "LE", "GT", "GE", "IntConst", 
  "FloatConst", "Identifier", "STRING", "WS", "LINE_COMMENT", "COMMENT"
};

dfa::Vocabulary SysYParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> SysYParser::_tokenNames;

SysYParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x2f, 0x179, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 
    0x1f, 0x9, 0x1f, 0x4, 0x20, 0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x3, 0x2, 
    0x3, 0x2, 0x6, 0x2, 0x45, 0xa, 0x2, 0xd, 0x2, 0xe, 0x2, 0x46, 0x3, 0x2, 
    0x3, 0x2, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x4d, 0xa, 0x3, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x7, 0x4, 0x54, 0xa, 0x4, 0xc, 0x4, 
    0xe, 0x4, 0x57, 0xb, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x7, 0x6, 0x62, 0xa, 0x6, 
    0xc, 0x6, 0xe, 0x6, 0x65, 0xb, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x7, 0x7, 0x6f, 0xa, 0x7, 
    0xc, 0x7, 0xe, 0x7, 0x72, 0xb, 0x7, 0x5, 0x7, 0x74, 0xa, 0x7, 0x3, 0x7, 
    0x5, 0x7, 0x77, 0xa, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x7, 
    0x8, 0x7d, 0xa, 0x8, 0xc, 0x8, 0xe, 0x8, 0x80, 0xb, 0x8, 0x3, 0x8, 0x3, 
    0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x7, 0x9, 0x89, 
    0xa, 0x9, 0xc, 0x9, 0xe, 0x9, 0x8c, 0xb, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x7, 0x9, 0x93, 0xa, 0x9, 0xc, 0x9, 0xe, 0x9, 
    0x96, 0xb, 0x9, 0x3, 0x9, 0x3, 0x9, 0x5, 0x9, 0x9a, 0xa, 0x9, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x7, 0xa, 0xa1, 0xa, 0xa, 0xc, 
    0xa, 0xe, 0xa, 0xa4, 0xb, 0xa, 0x5, 0xa, 0xa6, 0xa, 0xa, 0x3, 0xa, 0x5, 
    0xa, 0xa9, 0xa, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 
    0xaf, 0xa, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x7, 0xd, 0xb9, 0xa, 0xd, 0xc, 0xd, 0xe, 0xd, 
    0xbc, 0xb, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x7, 0xe, 0xc6, 0xa, 0xe, 0xc, 0xe, 0xe, 0xe, 
    0xc9, 0xb, 0xe, 0x5, 0xe, 0xcb, 0xa, 0xe, 0x3, 0xf, 0x3, 0xf, 0x7, 0xf, 
    0xcf, 0xa, 0xf, 0xc, 0xf, 0xe, 0xf, 0xd2, 0xb, 0xf, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0x10, 0x3, 0x10, 0x5, 0x10, 0xd8, 0xa, 0x10, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x5, 0x11, 0xe0, 0xa, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x5, 0x11, 0xfe, 0xa, 0x11, 0x3, 0x11, 0x5, 0x11, 
    0x101, 0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x7, 0x14, 0x10c, 0xa, 0x14, 
    0xc, 0x14, 0xe, 0x14, 0x10f, 0xb, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 
    0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x5, 0x15, 0x117, 0xa, 0x15, 0x3, 0x16, 
    0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 0x11f, 
    0xa, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 0x125, 
    0xa, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x7, 
    0x19, 0x12c, 0xa, 0x19, 0xc, 0x19, 0xe, 0x19, 0x12f, 0xb, 0x19, 0x3, 
    0x1a, 0x3, 0x1a, 0x5, 0x1a, 0x133, 0xa, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 
    0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x7, 0x1b, 0x13b, 0xa, 0x1b, 
    0xc, 0x1b, 0xe, 0x1b, 0x13e, 0xb, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
    0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x7, 0x1c, 0x146, 0xa, 0x1c, 0xc, 0x1c, 
    0xe, 0x1c, 0x149, 0xb, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 
    0x3, 0x1d, 0x3, 0x1d, 0x7, 0x1d, 0x151, 0xa, 0x1d, 0xc, 0x1d, 0xe, 0x1d, 
    0x154, 0xb, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x7, 0x1e, 0x15c, 0xa, 0x1e, 0xc, 0x1e, 0xe, 0x1e, 0x15f, 
    0xb, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 
    0x1f, 0x7, 0x1f, 0x167, 0xa, 0x1f, 0xc, 0x1f, 0xe, 0x1f, 0x16a, 0xb, 
    0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
    0x7, 0x20, 0x172, 0xa, 0x20, 0xc, 0x20, 0xe, 0x20, 0x175, 0xb, 0x20, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x2, 0x8, 0x34, 0x36, 0x38, 0x3a, 0x3c, 
    0x3e, 0x22, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 
    0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 
    0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x2, 0xa, 0x3, 
    0x2, 0x3, 0x5, 0x3, 0x2, 0x3, 0x6, 0x3, 0x2, 0x29, 0x2a, 0x4, 0x2, 0x1a, 
    0x1b, 0x1d, 0x1d, 0x3, 0x2, 0x1e, 0x20, 0x4, 0x2, 0x1a, 0x1a, 0x1d, 
    0x1d, 0x3, 0x2, 0x25, 0x28, 0x3, 0x2, 0x23, 0x24, 0x2, 0x185, 0x2, 0x44, 
    0x3, 0x2, 0x2, 0x2, 0x4, 0x4c, 0x3, 0x2, 0x2, 0x2, 0x6, 0x4e, 0x3, 0x2, 
    0x2, 0x2, 0x8, 0x5a, 0x3, 0x2, 0x2, 0x2, 0xa, 0x5c, 0x3, 0x2, 0x2, 0x2, 
    0xc, 0x76, 0x3, 0x2, 0x2, 0x2, 0xe, 0x78, 0x3, 0x2, 0x2, 0x2, 0x10, 
    0x99, 0x3, 0x2, 0x2, 0x2, 0x12, 0xa8, 0x3, 0x2, 0x2, 0x2, 0x14, 0xaa, 
    0x3, 0x2, 0x2, 0x2, 0x16, 0xb3, 0x3, 0x2, 0x2, 0x2, 0x18, 0xb5, 0x3, 
    0x2, 0x2, 0x2, 0x1a, 0xbd, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xcc, 0x3, 0x2, 
    0x2, 0x2, 0x1e, 0xd7, 0x3, 0x2, 0x2, 0x2, 0x20, 0x100, 0x3, 0x2, 0x2, 
    0x2, 0x22, 0x102, 0x3, 0x2, 0x2, 0x2, 0x24, 0x104, 0x3, 0x2, 0x2, 0x2, 
    0x26, 0x106, 0x3, 0x2, 0x2, 0x2, 0x28, 0x116, 0x3, 0x2, 0x2, 0x2, 0x2a, 
    0x118, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x124, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x126, 
    0x3, 0x2, 0x2, 0x2, 0x30, 0x128, 0x3, 0x2, 0x2, 0x2, 0x32, 0x132, 0x3, 
    0x2, 0x2, 0x2, 0x34, 0x134, 0x3, 0x2, 0x2, 0x2, 0x36, 0x13f, 0x3, 0x2, 
    0x2, 0x2, 0x38, 0x14a, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x155, 0x3, 0x2, 0x2, 
    0x2, 0x3c, 0x160, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x16b, 0x3, 0x2, 0x2, 0x2, 
    0x40, 0x176, 0x3, 0x2, 0x2, 0x2, 0x42, 0x45, 0x5, 0x4, 0x3, 0x2, 0x43, 
    0x45, 0x5, 0x14, 0xb, 0x2, 0x44, 0x42, 0x3, 0x2, 0x2, 0x2, 0x44, 0x43, 
    0x3, 0x2, 0x2, 0x2, 0x45, 0x46, 0x3, 0x2, 0x2, 0x2, 0x46, 0x44, 0x3, 
    0x2, 0x2, 0x2, 0x46, 0x47, 0x3, 0x2, 0x2, 0x2, 0x47, 0x48, 0x3, 0x2, 
    0x2, 0x2, 0x48, 0x49, 0x7, 0x2, 0x2, 0x3, 0x49, 0x3, 0x3, 0x2, 0x2, 
    0x2, 0x4a, 0x4d, 0x5, 0x6, 0x4, 0x2, 0x4b, 0x4d, 0x5, 0xe, 0x8, 0x2, 
    0x4c, 0x4a, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x4d, 
    0x5, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x4f, 0x7, 0x7, 0x2, 0x2, 0x4f, 0x50, 
    0x5, 0x8, 0x5, 0x2, 0x50, 0x55, 0x5, 0xa, 0x6, 0x2, 0x51, 0x52, 0x7, 
    0x15, 0x2, 0x2, 0x52, 0x54, 0x5, 0xa, 0x6, 0x2, 0x53, 0x51, 0x3, 0x2, 
    0x2, 0x2, 0x54, 0x57, 0x3, 0x2, 0x2, 0x2, 0x55, 0x53, 0x3, 0x2, 0x2, 
    0x2, 0x55, 0x56, 0x3, 0x2, 0x2, 0x2, 0x56, 0x58, 0x3, 0x2, 0x2, 0x2, 
    0x57, 0x55, 0x3, 0x2, 0x2, 0x2, 0x58, 0x59, 0x7, 0x16, 0x2, 0x2, 0x59, 
    0x7, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x5b, 0x9, 0x2, 0x2, 0x2, 0x5b, 0x9, 
    0x3, 0x2, 0x2, 0x2, 0x5c, 0x63, 0x7, 0x2b, 0x2, 0x2, 0x5d, 0x5e, 0x7, 
    0x11, 0x2, 0x2, 0x5e, 0x5f, 0x5, 0x40, 0x21, 0x2, 0x5f, 0x60, 0x7, 0x12, 
    0x2, 0x2, 0x60, 0x62, 0x3, 0x2, 0x2, 0x2, 0x61, 0x5d, 0x3, 0x2, 0x2, 
    0x2, 0x62, 0x65, 0x3, 0x2, 0x2, 0x2, 0x63, 0x61, 0x3, 0x2, 0x2, 0x2, 
    0x63, 0x64, 0x3, 0x2, 0x2, 0x2, 0x64, 0x66, 0x3, 0x2, 0x2, 0x2, 0x65, 
    0x63, 0x3, 0x2, 0x2, 0x2, 0x66, 0x67, 0x7, 0x19, 0x2, 0x2, 0x67, 0x68, 
    0x5, 0xc, 0x7, 0x2, 0x68, 0xb, 0x3, 0x2, 0x2, 0x2, 0x69, 0x77, 0x5, 
    0x40, 0x21, 0x2, 0x6a, 0x73, 0x7, 0x13, 0x2, 0x2, 0x6b, 0x70, 0x5, 0xc, 
    0x7, 0x2, 0x6c, 0x6d, 0x7, 0x15, 0x2, 0x2, 0x6d, 0x6f, 0x5, 0xc, 0x7, 
    0x2, 0x6e, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x72, 0x3, 0x2, 0x2, 0x2, 
    0x70, 0x6e, 0x3, 0x2, 0x2, 0x2, 0x70, 0x71, 0x3, 0x2, 0x2, 0x2, 0x71, 
    0x74, 0x3, 0x2, 0x2, 0x2, 0x72, 0x70, 0x3, 0x2, 0x2, 0x2, 0x73, 0x6b, 
    0x3, 0x2, 0x2, 0x2, 0x73, 0x74, 0x3, 0x2, 0x2, 0x2, 0x74, 0x75, 0x3, 
    0x2, 0x2, 0x2, 0x75, 0x77, 0x7, 0x14, 0x2, 0x2, 0x76, 0x69, 0x3, 0x2, 
    0x2, 0x2, 0x76, 0x6a, 0x3, 0x2, 0x2, 0x2, 0x77, 0xd, 0x3, 0x2, 0x2, 
    0x2, 0x78, 0x79, 0x5, 0x8, 0x5, 0x2, 0x79, 0x7e, 0x5, 0x10, 0x9, 0x2, 
    0x7a, 0x7b, 0x7, 0x15, 0x2, 0x2, 0x7b, 0x7d, 0x5, 0x10, 0x9, 0x2, 0x7c, 
    0x7a, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x80, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x7c, 
    0x3, 0x2, 0x2, 0x2, 0x7e, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x7f, 0x81, 0x3, 
    0x2, 0x2, 0x2, 0x80, 0x7e, 0x3, 0x2, 0x2, 0x2, 0x81, 0x82, 0x7, 0x16, 
    0x2, 0x2, 0x82, 0xf, 0x3, 0x2, 0x2, 0x2, 0x83, 0x8a, 0x7, 0x2b, 0x2, 
    0x2, 0x84, 0x85, 0x7, 0x11, 0x2, 0x2, 0x85, 0x86, 0x5, 0x40, 0x21, 0x2, 
    0x86, 0x87, 0x7, 0x12, 0x2, 0x2, 0x87, 0x89, 0x3, 0x2, 0x2, 0x2, 0x88, 
    0x84, 0x3, 0x2, 0x2, 0x2, 0x89, 0x8c, 0x3, 0x2, 0x2, 0x2, 0x8a, 0x88, 
    0x3, 0x2, 0x2, 0x2, 0x8a, 0x8b, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x9a, 0x3, 
    0x2, 0x2, 0x2, 0x8c, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x94, 0x7, 0x2b, 
    0x2, 0x2, 0x8e, 0x8f, 0x7, 0x11, 0x2, 0x2, 0x8f, 0x90, 0x5, 0x40, 0x21, 
    0x2, 0x90, 0x91, 0x7, 0x12, 0x2, 0x2, 0x91, 0x93, 0x3, 0x2, 0x2, 0x2, 
    0x92, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x93, 0x96, 0x3, 0x2, 0x2, 0x2, 0x94, 
    0x92, 0x3, 0x2, 0x2, 0x2, 0x94, 0x95, 0x3, 0x2, 0x2, 0x2, 0x95, 0x97, 
    0x3, 0x2, 0x2, 0x2, 0x96, 0x94, 0x3, 0x2, 0x2, 0x2, 0x97, 0x98, 0x7, 
    0x19, 0x2, 0x2, 0x98, 0x9a, 0x5, 0x12, 0xa, 0x2, 0x99, 0x83, 0x3, 0x2, 
    0x2, 0x2, 0x99, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x11, 0x3, 0x2, 0x2, 
    0x2, 0x9b, 0xa9, 0x5, 0x22, 0x12, 0x2, 0x9c, 0xa5, 0x7, 0x13, 0x2, 0x2, 
    0x9d, 0xa2, 0x5, 0x12, 0xa, 0x2, 0x9e, 0x9f, 0x7, 0x15, 0x2, 0x2, 0x9f, 
    0xa1, 0x5, 0x12, 0xa, 0x2, 0xa0, 0x9e, 0x3, 0x2, 0x2, 0x2, 0xa1, 0xa4, 
    0x3, 0x2, 0x2, 0x2, 0xa2, 0xa0, 0x3, 0x2, 0x2, 0x2, 0xa2, 0xa3, 0x3, 
    0x2, 0x2, 0x2, 0xa3, 0xa6, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa2, 0x3, 0x2, 
    0x2, 0x2, 0xa5, 0x9d, 0x3, 0x2, 0x2, 0x2, 0xa5, 0xa6, 0x3, 0x2, 0x2, 
    0x2, 0xa6, 0xa7, 0x3, 0x2, 0x2, 0x2, 0xa7, 0xa9, 0x7, 0x14, 0x2, 0x2, 
    0xa8, 0x9b, 0x3, 0x2, 0x2, 0x2, 0xa8, 0x9c, 0x3, 0x2, 0x2, 0x2, 0xa9, 
    0x13, 0x3, 0x2, 0x2, 0x2, 0xaa, 0xab, 0x5, 0x16, 0xc, 0x2, 0xab, 0xac, 
    0x7, 0x2b, 0x2, 0x2, 0xac, 0xae, 0x7, 0xf, 0x2, 0x2, 0xad, 0xaf, 0x5, 
    0x18, 0xd, 0x2, 0xae, 0xad, 0x3, 0x2, 0x2, 0x2, 0xae, 0xaf, 0x3, 0x2, 
    0x2, 0x2, 0xaf, 0xb0, 0x3, 0x2, 0x2, 0x2, 0xb0, 0xb1, 0x7, 0x10, 0x2, 
    0x2, 0xb1, 0xb2, 0x5, 0x1c, 0xf, 0x2, 0xb2, 0x15, 0x3, 0x2, 0x2, 0x2, 
    0xb3, 0xb4, 0x9, 0x3, 0x2, 0x2, 0xb4, 0x17, 0x3, 0x2, 0x2, 0x2, 0xb5, 
    0xba, 0x5, 0x1a, 0xe, 0x2, 0xb6, 0xb7, 0x7, 0x15, 0x2, 0x2, 0xb7, 0xb9, 
    0x5, 0x1a, 0xe, 0x2, 0xb8, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xbc, 0x3, 
    0x2, 0x2, 0x2, 0xba, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xba, 0xbb, 0x3, 0x2, 
    0x2, 0x2, 0xbb, 0x19, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xba, 0x3, 0x2, 0x2, 
    0x2, 0xbd, 0xbe, 0x5, 0x8, 0x5, 0x2, 0xbe, 0xca, 0x7, 0x2b, 0x2, 0x2, 
    0xbf, 0xc0, 0x7, 0x11, 0x2, 0x2, 0xc0, 0xc7, 0x7, 0x12, 0x2, 0x2, 0xc1, 
    0xc2, 0x7, 0x11, 0x2, 0x2, 0xc2, 0xc3, 0x5, 0x40, 0x21, 0x2, 0xc3, 0xc4, 
    0x7, 0x12, 0x2, 0x2, 0xc4, 0xc6, 0x3, 0x2, 0x2, 0x2, 0xc5, 0xc1, 0x3, 
    0x2, 0x2, 0x2, 0xc6, 0xc9, 0x3, 0x2, 0x2, 0x2, 0xc7, 0xc5, 0x3, 0x2, 
    0x2, 0x2, 0xc7, 0xc8, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xcb, 0x3, 0x2, 0x2, 
    0x2, 0xc9, 0xc7, 0x3, 0x2, 0x2, 0x2, 0xca, 0xbf, 0x3, 0x2, 0x2, 0x2, 
    0xca, 0xcb, 0x3, 0x2, 0x2, 0x2, 0xcb, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xcc, 
    0xd0, 0x7, 0x13, 0x2, 0x2, 0xcd, 0xcf, 0x5, 0x1e, 0x10, 0x2, 0xce, 0xcd, 
    0x3, 0x2, 0x2, 0x2, 0xcf, 0xd2, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xce, 0x3, 
    0x2, 0x2, 0x2, 0xd0, 0xd1, 0x3, 0x2, 0x2, 0x2, 0xd1, 0xd3, 0x3, 0x2, 
    0x2, 0x2, 0xd2, 0xd0, 0x3, 0x2, 0x2, 0x2, 0xd3, 0xd4, 0x7, 0x14, 0x2, 
    0x2, 0xd4, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd8, 0x5, 0x4, 0x3, 0x2, 
    0xd6, 0xd8, 0x5, 0x20, 0x11, 0x2, 0xd7, 0xd5, 0x3, 0x2, 0x2, 0x2, 0xd7, 
    0xd6, 0x3, 0x2, 0x2, 0x2, 0xd8, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xda, 
    0x5, 0x26, 0x14, 0x2, 0xda, 0xdb, 0x7, 0x19, 0x2, 0x2, 0xdb, 0xdc, 0x5, 
    0x22, 0x12, 0x2, 0xdc, 0xdd, 0x7, 0x16, 0x2, 0x2, 0xdd, 0x101, 0x3, 
    0x2, 0x2, 0x2, 0xde, 0xe0, 0x5, 0x22, 0x12, 0x2, 0xdf, 0xde, 0x3, 0x2, 
    0x2, 0x2, 0xdf, 0xe0, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xe1, 0x3, 0x2, 0x2, 
    0x2, 0xe1, 0x101, 0x7, 0x16, 0x2, 0x2, 0xe2, 0x101, 0x5, 0x1c, 0xf, 
    0x2, 0xe3, 0xe4, 0x7, 0x9, 0x2, 0x2, 0xe4, 0xe5, 0x7, 0xf, 0x2, 0x2, 
    0xe5, 0xe6, 0x5, 0x24, 0x13, 0x2, 0xe6, 0xe7, 0x7, 0x10, 0x2, 0x2, 0xe7, 
    0xe8, 0x5, 0x20, 0x11, 0x2, 0xe8, 0x101, 0x3, 0x2, 0x2, 0x2, 0xe9, 0xea, 
    0x7, 0x9, 0x2, 0x2, 0xea, 0xeb, 0x7, 0xf, 0x2, 0x2, 0xeb, 0xec, 0x5, 
    0x24, 0x13, 0x2, 0xec, 0xed, 0x7, 0x10, 0x2, 0x2, 0xed, 0xee, 0x5, 0x20, 
    0x11, 0x2, 0xee, 0xef, 0x7, 0xa, 0x2, 0x2, 0xef, 0xf0, 0x5, 0x20, 0x11, 
    0x2, 0xf0, 0x101, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xf2, 0x7, 0xc, 0x2, 0x2, 
    0xf2, 0xf3, 0x7, 0xf, 0x2, 0x2, 0xf3, 0xf4, 0x5, 0x24, 0x13, 0x2, 0xf4, 
    0xf5, 0x7, 0x10, 0x2, 0x2, 0xf5, 0xf6, 0x5, 0x20, 0x11, 0x2, 0xf6, 0x101, 
    0x3, 0x2, 0x2, 0x2, 0xf7, 0xf8, 0x7, 0xd, 0x2, 0x2, 0xf8, 0x101, 0x7, 
    0x16, 0x2, 0x2, 0xf9, 0xfa, 0x7, 0xe, 0x2, 0x2, 0xfa, 0x101, 0x7, 0x16, 
    0x2, 0x2, 0xfb, 0xfd, 0x7, 0x8, 0x2, 0x2, 0xfc, 0xfe, 0x5, 0x22, 0x12, 
    0x2, 0xfd, 0xfc, 0x3, 0x2, 0x2, 0x2, 0xfd, 0xfe, 0x3, 0x2, 0x2, 0x2, 
    0xfe, 0xff, 0x3, 0x2, 0x2, 0x2, 0xff, 0x101, 0x7, 0x16, 0x2, 0x2, 0x100, 
    0xd9, 0x3, 0x2, 0x2, 0x2, 0x100, 0xdf, 0x3, 0x2, 0x2, 0x2, 0x100, 0xe2, 
    0x3, 0x2, 0x2, 0x2, 0x100, 0xe3, 0x3, 0x2, 0x2, 0x2, 0x100, 0xe9, 0x3, 
    0x2, 0x2, 0x2, 0x100, 0xf1, 0x3, 0x2, 0x2, 0x2, 0x100, 0xf7, 0x3, 0x2, 
    0x2, 0x2, 0x100, 0xf9, 0x3, 0x2, 0x2, 0x2, 0x100, 0xfb, 0x3, 0x2, 0x2, 
    0x2, 0x101, 0x21, 0x3, 0x2, 0x2, 0x2, 0x102, 0x103, 0x5, 0x36, 0x1c, 
    0x2, 0x103, 0x23, 0x3, 0x2, 0x2, 0x2, 0x104, 0x105, 0x5, 0x3e, 0x20, 
    0x2, 0x105, 0x25, 0x3, 0x2, 0x2, 0x2, 0x106, 0x10d, 0x7, 0x2b, 0x2, 
    0x2, 0x107, 0x108, 0x7, 0x11, 0x2, 0x2, 0x108, 0x109, 0x5, 0x22, 0x12, 
    0x2, 0x109, 0x10a, 0x7, 0x12, 0x2, 0x2, 0x10a, 0x10c, 0x3, 0x2, 0x2, 
    0x2, 0x10b, 0x107, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10f, 0x3, 0x2, 0x2, 
    0x2, 0x10d, 0x10b, 0x3, 0x2, 0x2, 0x2, 0x10d, 0x10e, 0x3, 0x2, 0x2, 
    0x2, 0x10e, 0x27, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x10d, 0x3, 0x2, 0x2, 0x2, 
    0x110, 0x111, 0x7, 0xf, 0x2, 0x2, 0x111, 0x112, 0x5, 0x22, 0x12, 0x2, 
    0x112, 0x113, 0x7, 0x10, 0x2, 0x2, 0x113, 0x117, 0x3, 0x2, 0x2, 0x2, 
    0x114, 0x117, 0x5, 0x26, 0x14, 0x2, 0x115, 0x117, 0x5, 0x2a, 0x16, 0x2, 
    0x116, 0x110, 0x3, 0x2, 0x2, 0x2, 0x116, 0x114, 0x3, 0x2, 0x2, 0x2, 
    0x116, 0x115, 0x3, 0x2, 0x2, 0x2, 0x117, 0x29, 0x3, 0x2, 0x2, 0x2, 0x118, 
    0x119, 0x9, 0x4, 0x2, 0x2, 0x119, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x125, 
    0x5, 0x28, 0x15, 0x2, 0x11b, 0x11c, 0x7, 0x2b, 0x2, 0x2, 0x11c, 0x11e, 
    0x7, 0xf, 0x2, 0x2, 0x11d, 0x11f, 0x5, 0x30, 0x19, 0x2, 0x11e, 0x11d, 
    0x3, 0x2, 0x2, 0x2, 0x11e, 0x11f, 0x3, 0x2, 0x2, 0x2, 0x11f, 0x120, 
    0x3, 0x2, 0x2, 0x2, 0x120, 0x125, 0x7, 0x10, 0x2, 0x2, 0x121, 0x122, 
    0x5, 0x2e, 0x18, 0x2, 0x122, 0x123, 0x5, 0x2c, 0x17, 0x2, 0x123, 0x125, 
    0x3, 0x2, 0x2, 0x2, 0x124, 0x11a, 0x3, 0x2, 0x2, 0x2, 0x124, 0x11b, 
    0x3, 0x2, 0x2, 0x2, 0x124, 0x121, 0x3, 0x2, 0x2, 0x2, 0x125, 0x2d, 0x3, 
    0x2, 0x2, 0x2, 0x126, 0x127, 0x9, 0x5, 0x2, 0x2, 0x127, 0x2f, 0x3, 0x2, 
    0x2, 0x2, 0x128, 0x12d, 0x5, 0x32, 0x1a, 0x2, 0x129, 0x12a, 0x7, 0x15, 
    0x2, 0x2, 0x12a, 0x12c, 0x5, 0x32, 0x1a, 0x2, 0x12b, 0x129, 0x3, 0x2, 
    0x2, 0x2, 0x12c, 0x12f, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x12b, 0x3, 0x2, 
    0x2, 0x2, 0x12d, 0x12e, 0x3, 0x2, 0x2, 0x2, 0x12e, 0x31, 0x3, 0x2, 0x2, 
    0x2, 0x12f, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x130, 0x133, 0x5, 0x22, 0x12, 
    0x2, 0x131, 0x133, 0x7, 0x2c, 0x2, 0x2, 0x132, 0x130, 0x3, 0x2, 0x2, 
    0x2, 0x132, 0x131, 0x3, 0x2, 0x2, 0x2, 0x133, 0x33, 0x3, 0x2, 0x2, 0x2, 
    0x134, 0x135, 0x8, 0x1b, 0x1, 0x2, 0x135, 0x136, 0x5, 0x2c, 0x17, 0x2, 
    0x136, 0x13c, 0x3, 0x2, 0x2, 0x2, 0x137, 0x138, 0xc, 0x3, 0x2, 0x2, 
    0x138, 0x139, 0x9, 0x6, 0x2, 0x2, 0x139, 0x13b, 0x5, 0x2c, 0x17, 0x2, 
    0x13a, 0x137, 0x3, 0x2, 0x2, 0x2, 0x13b, 0x13e, 0x3, 0x2, 0x2, 0x2, 
    0x13c, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x13d, 0x3, 0x2, 0x2, 0x2, 
    0x13d, 0x35, 0x3, 0x2, 0x2, 0x2, 0x13e, 0x13c, 0x3, 0x2, 0x2, 0x2, 0x13f, 
    0x140, 0x8, 0x1c, 0x1, 0x2, 0x140, 0x141, 0x5, 0x34, 0x1b, 0x2, 0x141, 
    0x147, 0x3, 0x2, 0x2, 0x2, 0x142, 0x143, 0xc, 0x3, 0x2, 0x2, 0x143, 
    0x144, 0x9, 0x7, 0x2, 0x2, 0x144, 0x146, 0x5, 0x34, 0x1b, 0x2, 0x145, 
    0x142, 0x3, 0x2, 0x2, 0x2, 0x146, 0x149, 0x3, 0x2, 0x2, 0x2, 0x147, 
    0x145, 0x3, 0x2, 0x2, 0x2, 0x147, 0x148, 0x3, 0x2, 0x2, 0x2, 0x148, 
    0x37, 0x3, 0x2, 0x2, 0x2, 0x149, 0x147, 0x3, 0x2, 0x2, 0x2, 0x14a, 0x14b, 
    0x8, 0x1d, 0x1, 0x2, 0x14b, 0x14c, 0x5, 0x36, 0x1c, 0x2, 0x14c, 0x152, 
    0x3, 0x2, 0x2, 0x2, 0x14d, 0x14e, 0xc, 0x3, 0x2, 0x2, 0x14e, 0x14f, 
    0x9, 0x8, 0x2, 0x2, 0x14f, 0x151, 0x5, 0x36, 0x1c, 0x2, 0x150, 0x14d, 
    0x3, 0x2, 0x2, 0x2, 0x151, 0x154, 0x3, 0x2, 0x2, 0x2, 0x152, 0x150, 
    0x3, 0x2, 0x2, 0x2, 0x152, 0x153, 0x3, 0x2, 0x2, 0x2, 0x153, 0x39, 0x3, 
    0x2, 0x2, 0x2, 0x154, 0x152, 0x3, 0x2, 0x2, 0x2, 0x155, 0x156, 0x8, 
    0x1e, 0x1, 0x2, 0x156, 0x157, 0x5, 0x38, 0x1d, 0x2, 0x157, 0x15d, 0x3, 
    0x2, 0x2, 0x2, 0x158, 0x159, 0xc, 0x3, 0x2, 0x2, 0x159, 0x15a, 0x9, 
    0x9, 0x2, 0x2, 0x15a, 0x15c, 0x5, 0x38, 0x1d, 0x2, 0x15b, 0x158, 0x3, 
    0x2, 0x2, 0x2, 0x15c, 0x15f, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15b, 0x3, 
    0x2, 0x2, 0x2, 0x15d, 0x15e, 0x3, 0x2, 0x2, 0x2, 0x15e, 0x3b, 0x3, 0x2, 
    0x2, 0x2, 0x15f, 0x15d, 0x3, 0x2, 0x2, 0x2, 0x160, 0x161, 0x8, 0x1f, 
    0x1, 0x2, 0x161, 0x162, 0x5, 0x3a, 0x1e, 0x2, 0x162, 0x168, 0x3, 0x2, 
    0x2, 0x2, 0x163, 0x164, 0xc, 0x3, 0x2, 0x2, 0x164, 0x165, 0x7, 0x21, 
    0x2, 0x2, 0x165, 0x167, 0x5, 0x3a, 0x1e, 0x2, 0x166, 0x163, 0x3, 0x2, 
    0x2, 0x2, 0x167, 0x16a, 0x3, 0x2, 0x2, 0x2, 0x168, 0x166, 0x3, 0x2, 
    0x2, 0x2, 0x168, 0x169, 0x3, 0x2, 0x2, 0x2, 0x169, 0x3d, 0x3, 0x2, 0x2, 
    0x2, 0x16a, 0x168, 0x3, 0x2, 0x2, 0x2, 0x16b, 0x16c, 0x8, 0x20, 0x1, 
    0x2, 0x16c, 0x16d, 0x5, 0x3c, 0x1f, 0x2, 0x16d, 0x173, 0x3, 0x2, 0x2, 
    0x2, 0x16e, 0x16f, 0xc, 0x3, 0x2, 0x2, 0x16f, 0x170, 0x7, 0x22, 0x2, 
    0x2, 0x170, 0x172, 0x5, 0x3c, 0x1f, 0x2, 0x171, 0x16e, 0x3, 0x2, 0x2, 
    0x2, 0x172, 0x175, 0x3, 0x2, 0x2, 0x2, 0x173, 0x171, 0x3, 0x2, 0x2, 
    0x2, 0x173, 0x174, 0x3, 0x2, 0x2, 0x2, 0x174, 0x3f, 0x3, 0x2, 0x2, 0x2, 
    0x175, 0x173, 0x3, 0x2, 0x2, 0x2, 0x176, 0x177, 0x5, 0x36, 0x1c, 0x2, 
    0x177, 0x41, 0x3, 0x2, 0x2, 0x2, 0x26, 0x44, 0x46, 0x4c, 0x55, 0x63, 
    0x70, 0x73, 0x76, 0x7e, 0x8a, 0x94, 0x99, 0xa2, 0xa5, 0xa8, 0xae, 0xba, 
    0xc7, 0xca, 0xd0, 0xd7, 0xdf, 0xfd, 0x100, 0x10d, 0x116, 0x11e, 0x124, 
    0x12d, 0x132, 0x13c, 0x147, 0x152, 0x15d, 0x168, 0x173, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

SysYParser::Initializer SysYParser::_init;
