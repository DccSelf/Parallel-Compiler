
// Generated from SysY.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"




class  SysYLexer : public antlr4::Lexer {
public:
  enum {
    Int = 1, Float = 2, Tensor = 3, Void = 4, Const = 5, Return = 6, If = 7, 
    Else = 8, For = 9, While = 10, Break = 11, Continue = 12, Lparen = 13, 
    Rparen = 14, Lbrkt = 15, Rbrkt = 16, Lbrace = 17, Rbrace = 18, Comma = 19, 
    Semicolon = 20, Question = 21, Colon = 22, Assign = 23, Minus = 24, 
    Exclamation = 25, Tilde = 26, Addition = 27, Multiplication = 28, Division = 29, 
    Modulo = 30, LAND = 31, LOR = 32, EQ = 33, NEQ = 34, LT = 35, LE = 36, 
    GT = 37, GE = 38, IntConst = 39, FloatConst = 40, Identifier = 41, STRING = 42, 
    WS = 43, LINE_COMMENT = 44, COMMENT = 45
  };

  SysYLexer(antlr4::CharStream *input);
  ~SysYLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

