require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('break-oov-words-into-charseq.lua')

local options = {
  {'-vocab', '', 'Vocabulary File', {valid=onmt.utils.ExtendedCmdLine.fileNullOrExists}},
  {'-data', '', 'Monolingual Corpus', {valid=onmt.utils.ExtendedCmdLine.fileExists}},
  {'-save_data', '', 'Output Monolingual Corpus; All OOV words will be splitted into character sequence',
    {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-vocab_size', '50000', 'Comma-separated list of target vocabularies size: word[,feat1,feat2,...].',
    {valid=onmt.utils.ExtendedCmdLine.listUInt}},
  {'-features_vocabs_prefix', '',      [[Path prefix to existing features vocabularies.]]},
}

cmd:setCmdLineOptions(options, 'Breaking OOVs into sequence of its constituent characters')
onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function isValid(sent)
  return #sent > 0
end

local function main()
  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local Vocabulary = onmt.data.Vocabulary

  local dicts = {}
  dicts.src = Vocabulary.init('train',
                                   opt.data,
                                   opt.vocab,
                                   opt.vocab_size,
                                   opt.features_vocabs_prefix,
                                   function(s) return isValid(s) end)

  _G.logger:info('Re-scanning a corpus file and Convert OOV words')

  local unicode = require('tools.utils.unicode')
  local reader = onmt.utils.FileReader.new(opt.data)
  local new_corpusfile = assert(io.open(opt.save_data, 'w'))

  local linecount = 1
  while true do
    local tokens = reader:next()
    linecount = linecount + 1

    if tokens == nil then
      break
    end

    if (linecount % 100000) == 0 then
      _G.logger:info('... ' .. linecount .. ' sentences processed.')
    end

    if isValid(tokens) then
      local words, _ = onmt.utils.Features.extract(tokens)

      local idxed_words = dicts.src.words:convertToIdx(words, onmt.Constants.UNK_WORD)

      for i = 1, idxed_words:storage():size() do
        local a_word = words[i]
        if idxed_words[i] == onmt.Constants.UNK then
          local new_a_word = ''
          -- break into character sequence with some markers
          local t = 1
          for _, c, _, nextc in unicode.utf8_iter(a_word) do
            if t == 1 then
              new_a_word = '<B>' .. c
            elseif nextc == nil then
              new_a_word = new_a_word .. ' <E>' .. c
            else
              new_a_word = new_a_word .. ' <M>' .. c
            end
            t = t + 1
          end
          if i > 1 then
            new_corpusfile:write(' ' .. new_a_word)
          else
            new_corpusfile:write(new_a_word)
          end
        else
          -- just save a word
          if i > 1 then
            new_corpusfile:write(' ' .. a_word)
          else
            new_corpusfile:write(a_word)
          end
        end
      end
      -- write <LF> into opt.save_data
      new_corpusfile:write('\n')
    else
      _G.logger:warning('0-length line detected. line: ' .. linecount)
    end
  end

  reader:close()
  _G.logger:info('Mixed Word/Character Tokenization completed successfully.')
end

main()

