require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('translate.lua')

local options = {
  {'-src', '', [[Source sequence to decode (one line per sequence)]],
               {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-tgt', '', [[True target sequence (optional)]]},
  {'-output', 'pred.txt', [[Path to output the predictions (each line will be the decoded sequence)]]}
}

cmd:setCmdLineOptions(options, 'Data')

onmt.translate.Translator.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

cmd:option('-time', false, [[Measure batch translation time]])

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.2f, " .. name .. " PPL: %.2f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local id_digit_align_enable = true

-- digit placeholder 추출 메서드
-- 입력 - 입력 텍스트
-- 출력 - placeholder로 수정된 입력 텍스트, 각 placeholder id table
function digit_placeholder_prep(input_text)
  local num_idx = 1
  local num_tab = {}
  local lrexonig = require 'rex_onig'	-- luarocks install lrexlib-oniguruma
  while true do
	-- 한국어 월 앞의 숫자는 매칭되지 않게 한다.
	-- sidx, eidx = lrexonig.find(input_text, "(?<!_)(?>[0-9]+)")
	local sidx, eidx = lrexonig.find(input_text, "(?<!_)(?>[0-9]+)(?! 월)")
	if sidx == nil then 
		break 
	end
	local a_found_value = input_text:sub(sidx, eidx)
	
	-- 숫자 1일 경우는 무시하도록 한다. 성수 일치 등의 문제로 인함
	if a_found_value == "1" then 
		input_text = input_text:sub(1, sidx-1) .. "__JUST!ONE!_" .. input_text:sub(eidx+1, #input_text)
		goto continue 
	end

	num_tab[num_idx] = a_found_value

	if id_digit_align_enable == true then
		input_text = input_text:sub(1, sidx-1) .. "__digit_" .. tostring(num_idx) .. " " .. input_text:sub(eidx+1, #input_text) 
		-- input_text = lrexonig.gsub(input_text, "(?<!_)(?[0-9]+", "__digit_" .. tostring(num_idx) .. " ", 1)
	else
		input_text = lrexonig.gsub(input_text, "(?<!_)[0-9]+", "__digit_", 1)
	end
	num_idx = num_idx + 1
	::continue::
  end

  -- recover single number "1"
  input_text = input_text:gsub("__JUST!ONE!_", "1")
  input_text = input_text:gsub("  ", " ")

  return input_text, num_tab
end


local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  local srcBatch = {}

  local goldReader
  local goldBatch

  local withGoldScore = opt.tgt:len() > 0

  if withGoldScore then
    goldReader = onmt.utils.FileReader.new(opt.tgt)
    goldBatch = {}
  end

  local translator = onmt.translate.Translator.new(opt)

  local outFile = io.open(opt.output, 'w')

  local sentId = 1
  local batchId = 1

  local predScoreTotal = 0
  local predWordsTotal = 0
  local goldScoreTotal = 0
  local goldWordsTotal = 0

  local timer
  if opt.time then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  -- Variables for recover digit placeholder --
  local mySentId = 1
  local numTabSentId = 1
  local global_num_tab = {}

  while true do
    local srcTokens = srcReader:next()

    -- 여기서 전처리 먼저 수행 - digit로 변경
    -- 먼저 token을 string으로 복원
    if srcTokens ~= nil then
      local myInputSent = table.concat(srcTokens, " ")
      local myModifiedSent, num_tab = digit_placeholder_prep(myInputSent)
      global_num_tab[mySentId] = num_tab   
      --print ('mySentId: ' .. tostring(mySentId))
      --print ('myModifiedSent: ' .. myModifiedSent)
      mySentId = mySentId + 1

      -- 다시 token으로 분리
      srcTokens = {}
      for word in myModifiedSent:gmatch'([^%s]+)' do
        table.insert(srcTokens, word)
      end
    end

    local goldTokens
    if withGoldScore then
      goldTokens = goldReader:next()
    end

    if srcTokens ~= nil then
      table.insert(srcBatch, translator:buildInput(srcTokens))

      if withGoldScore then
        table.insert(goldBatch, translator:buildInput(goldTokens))
      end
    elseif #srcBatch == 0 then
      break
    end

    if srcTokens == nil or #srcBatch == opt.batch_size then
      if opt.time then
        timer:resume()
      end

      local results = translator:translate(srcBatch, goldBatch)

      if opt.time then
        timer:stop()
      end

      for b = 1, #results do
        if (#srcBatch[b].words == 0) then
          _G.logger:warning('Line ' .. sentId .. ' is empty.')
          outFile:write('\n')
        else
          _G.logger:info('SENT %d: %s', sentId, translator:buildOutput(srcBatch[b]))

          if withGoldScore then
            _G.logger:info('GOLD %d: %s', sentId, translator:buildOutput(goldBatch[b]), results[b].goldScore)
            _G.logger:info("GOLD SCORE: %.2f", results[b].goldScore)
            goldScoreTotal = goldScoreTotal + results[b].goldScore
            goldWordsTotal = goldWordsTotal + #goldBatch[b].words
          end

          for n = 1, #results[b].preds do
            local sentence = translator:buildOutput(results[b].preds[n])

            local srcSent = translator:buildOutput(srcBatch[b])
            local predSent = sentence 

            -- 여기서 복원 --
            local recov_num_idx = 0
            while true do
              if id_digit_align_enable == true then
                -- 원래 __digit_[0-9]+ 가 되어야 하나, 영어쪽 corpus 수정 실패로 인해 임시로 수정. 10개는 넘어가지 않을 것이다.
                local sidx, eidx = predSent:find("__digit_[0-9]")
                if sidx == nil then break end
                local an_id = predSent:sub(sidx+8, eidx)
                --print ('an_id:' .. tostring(an_id))
                -- 숫자 다음에 0이 오면 공백을 제거
                -- print (#predSent)
                if eidx+2 < #predSent then
                  local next_wrd = predSent:sub(eidx+1, eidx+2)
                  --print ('next_wrd: [' .. next_wrd .. ']')
                  if next_wrd == " 0" then
                  predSent = predSent:sub(1, eidx) .. predSent:sub(eidx+2)
                  --print ('new res: ' .. predSent)
                  end
                end

                --print('current numtab:' .. tostring(b+(#predBatch*(batchId-1))))
                if global_num_tab[b+(#predBatch*(batchId-1))] == nil or global_num_tab[b+(#predBatch*(batchId-1))][tonumber(an_id)] == nil then
                  info[b].score = -150.0
                  break
                else
                  predSent = predSent:sub(1, sidx-1) .. global_num_tab[b+(#predBatch*(batchId-1))][tonumber(an_id)] .. predSent:sub(eidx + 1)
                end

              else
                sidx, eidx = predSent:find("__digit_")
                if sidx == nil then break end
                predSent = predSent:gsub("__digit_", global_num_tab[b+(#predBatch*(batchId-1))][recov_num_idx], 1)
              end
              recov_num_idx = recov_num_idx + 1
            end
            -- 다시 sentence에 결과물 복제 후 반환
            sentence = predSent

            if n == 1 then
              outFile:write(sentence .. '\n')
              predScoreTotal = predScoreTotal + results[b].preds[n].score
              predWordsTotal = predWordsTotal + #results[b].preds[n].words

              if #results[b].preds > 1 then
                _G.logger:info('')
                _G.logger:info('BEST HYP:')
              end
            end

            if #results[b].preds > 1 then
              _G.logger:info("[%.2f] %s", results[b].preds[n].score, sentence)
            else
              _G.logger:info("PRED %d: %s", sentId, sentence)
              _G.logger:info("PRED SCORE: %.2f", results[b].preds[n].score)
            end
          end
        end

        _G.logger:info('')
        sentId = sentId + 1
      end

      if srcTokens == nil then
        break
      end

      batchId = batchId + 1
      srcBatch = {}
      if withGoldScore then
        goldBatch = {}
      end
      collectgarbage()
    end
  end

  if opt.time then
    local time = timer:time()
    local sentenceCount = sentId-1
    _G.logger:info("Average sentence translation time (in seconds):\n")
    _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  end

  reportScore('PRED', predScoreTotal, predWordsTotal)

  if withGoldScore then
    reportScore('GOLD', goldScoreTotal, goldWordsTotal)
  end

  outFile:close()
  _G.logger:shutDown()
end

main()
