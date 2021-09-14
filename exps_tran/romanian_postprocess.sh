#! /bin/bash

ROOT=/home/chuntinz/tir5/tools/mosesdecoder
ro_post_process () {
  sys=$1
  ref=$2
  export MOSES_PATH=$ROOT
  REPLACE_UNICODE_PUNCT=$MOSES_PATH/scripts/tokenizer/replace-unicode-punctuation.perl
  NORM_PUNC=$MOSES_PATH/scripts/tokenizer/normalize-punctuation.perl
  REM_NON_PRINT_CHAR=$MOSES_PATH/scripts/tokenizer/remove-non-printing-char.perl
  REMOVE_DIACRITICS=$MOSES_PATH/wmt16-scripts/preprocess/remove-diacritics.py
  NORMALIZE_ROMANIAN=$MOSES_PATH/wmt16-scripts/preprocess/normalise-romanian.py
  TOKENIZER=$MOSES_PATH/scripts/tokenizer/tokenizer.perl

  lang=ro
  for file in $sys $ref; do
    cat $file \
    | $REPLACE_UNICODE_PUNCT \
    | $NORM_PUNC -l $lang \
    | $REM_NON_PRINT_CHAR \
    | $NORMALIZE_ROMANIAN \
    | $REMOVE_DIACRITICS \
    | $TOKENIZER -no-escape -l $lang \
    > $(basename $file).tok
  done
  # compute BLEU
  cat $(basename $sys).tok | sacrebleu -tok none -s none -b $(basename $ref).tok
}


ro_post_process ${1} ${2}