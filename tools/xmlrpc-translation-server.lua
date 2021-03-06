--[[
  A Simple XML-RPC server to serve OpenNMT result with Mosesserver style Parameters.
  Created by jhshin, ETRI NLP Team. 2017/02/23

  for xmlrpc server, we need to install luaxmlrpc, xavante, wsapi-xavante
  # sudo apt-get install libonig-dev
  $ luarocks install xavante; luarocks install wsapi-xavante; 
    luarocks install luaxmlrpc; 
    luarocks install lrexlib-oniguruma ONIG_LIBDIR=/usr/lib/x86_64-linux-gnu/

  * 2017/02/23
    - updated to OpenNMT v0.4 
]]

local DEBUG_PRINT = true

local xavante = require 'xavante'
local xavante_wsapi_inst = require 'wsapi.xavante'
local wsapi_request = require 'wsapi.request'
local xmlrpc = require 'xmlrpc'

local id_digit_align_enable = true 	-- we need always true, do not set to false, or output result will be bad alignments.
local lrexonig = require 'rex_onig'	-- luarocks install lrexlib-oniguruma

require 'xavante.httpd'

require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('xmlrpc_translation_server.lua')

-- translator를 밖으로 빼놓음
local translator

-- settings
local is_mixed = false

onmt.translate.Translator.declareOpts(cmd)

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text("")
cmd:text("** Server options **")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])
cmd:option('-host', '*', [[Host to run the server on]])
cmd:option('-port', '8020', [[Port to run the server on]])
cmd:option('-extra_digit', false, [[If it enabled, comma between numerics will be omitted.]])
cmd:option('-mixed', false, [[If it enabled, all OOVs will be splitted into consequence of its characters.]])
cmd:option('-bpe_model', '', [[Byte-pair encoding scheme based tokenization]])
cmd:option('-use_opennmt_joiner', false, [[If it enabled, use opennmt's joiner instead B/M/E symbols.]])

-- XML-RPC WSAPI Handler
function wsapi_handler(wsapi_env)
  local headers = { ["Content-type"] = "text/xml" }
  local req = wsapi_request.new(wsapi_env)
  local method, arg_table = xmlrpc.srvDecode(req.POST.post_data)
  --[[ arg_table의 경우, { 1: { "text" = value, "..." = "..." } } 와 같이
      들어오기 때문에, [1]로 한번 unpacking을 해 줘야 한다.
  ]]
  local func = xmlrpc.dispatch(method)
  local result = { pcall(func, arg_table[1]["text"], is_mixed) }
  --print ('result table: ', result)
  local is_ok = result[1]	-- pcall의 처리 결과가 나온다
  local final_result = {}

  if not is_ok then
    print('err: ' .. result[2])
    result = { code = 3, message = result[2] }
  else
    final_result["text"] = "<![CDATA[" .. result[2]:gsub("&", "%%26") .. "]]>";
    final_result["totalScore"] = result[3]
    final_result["normedScore"] = result[4]
  end

  local r = xmlrpc.srvEncode(final_result, not is_ok)
  --print(r)
  headers["Content-length"] = tostring(#r)
  local function xmlrpc_reply(wsapi_env)
    coroutine.yield(r)
  end

  return 200, headers, coroutine.wrap(xmlrpc_reply)
end

-- XML-RPC exported function lists
xmlrpc_exports = {}

function digit_placeholder_prep(input_text)
  local num_idx = 1
  local num_tab = {}
  local sidx = 0
  local eidx = 0

  -- 전화번호를 011 - 4573 - 1234 와 같이 떨어트리는 경우를 위해, 연속한 것을 붙이도록 한다.
  -- input_text = lrexonig.gsub(input_text, "(?<=[0-9]) - (?=[0-9])", "-")

  while true do
	-- 한국어 월 앞의 숫자는 매칭되지 않게 한다.
	-- sidx, eidx = lrexonig.find(input_text, "(?<!_)(?>[0-9]+)")
	if eidx ~= 0 then
          eidx = sidx + 10
        end
	--sidx, eidx = lrexonig.find(input_text, "(?<![_,.])[0-9][0-9.,-]*(?! 월)")  -- 아직 쓰지 말것. . , merging이 똑바로 안되어 있다.
	sidx, eidx = lrexonig.find(input_text, "(?<!_)(?>[0-9]+)(?! 월)")    -- old style (e.g. ke-25m-digit)
	if sidx == nil then 
		break 
	end
	local a_found_value = input_text:sub(sidx, eidx)
	
	-- 숫자 1일 경우는 무시하도록 한다. 성수 일치 등의 문제로 인함
	if a_found_value == "1" then 
		input_text = input_text:sub(1, sidx-1) .. "__JUST!ONE!_" .. input_text:sub(eidx+1, #input_text)
		goto continue 
	end
	print (num_idx .. ':' .. a_found_value)

	num_tab[num_idx] = a_found_value

	if id_digit_align_enable == true then
		--input_text = input_text:sub(1, sidx-1) .. "__digit_" .. tostring(num_idx) .. "__ " .. input_text:sub(eidx+1, #input_text) 
		input_text = input_text:sub(1, sidx-1) .. "__digit_" .. tostring(num_idx) .. " " .. input_text:sub(eidx+1, #input_text) 
	else
		input_text = lrexonig.gsub(input_text, "(?<!_)[0-9]+", "__digit_", 1)
	end
	--print('intermediate - ' .. input_text)
	num_idx = num_idx + 1
	if num_idx > 9 then
		break
	end
	::continue::
  end

  -- recover single number "1"
  input_text = input_text:gsub("__JUST!ONE!_", "1")
  input_text = input_text:gsub("  ", " ")
  
  print('modified input: ' .. input_text)

  return input_text, num_tab
end

function break_oov_into_chars(input_word)
  local unicode = require('tools.utils.unicode')
  local new_a_word = ''
  --print (input_word)
  local t = 1
  for _, c, _, nextc in unicode.utf8_iter(input_word) do
    if is_use_opennmt_joiner then
      if t ~= 1 then
        new_a_word = new_a_word .. ' ' .. require('tools.utils.separators').joiner_marker .. c
      else
        new_a_word = c
      end
    else
      --print(c)
      if t == 1 then
        new_a_word = '<B>' .. c
      elseif nextc == nil then
        new_a_word = new_a_word .. ' <E>' .. c
      else
        new_a_word = new_a_word .. ' <M>' .. c
      end
    end
    t = t + 1
  end
  return new_a_word
end

-- translate method
--[[ 
      Each XML-RPC procedures returns pcall result(true/false), return of procedures 1, return of procedures 2 ... as a lua table.
  ]]
function xmlrpc_exports.translate(input_text, is_mixed_param)
  print('Start Translate..')
  local unk_penalty = 2 
  local batch = {}

  -- Input text의 숫자를 읽어들여, 이를 placeholder로 바꿔주도록 한다.
  local pp_input_text, num_tab
  if _G.bpe ~= nil and _G.tokenizer ~= nil then
    local separators = require('tools.utils.separators')
    local bpeopt = {
      joiner_annotate = true,
      joiner = separators.joiner_marker,
      joiner_new = false
    }
    local tmpTokens = _G.tokenizer.tokenize(bpeopt, input_text, _G.bpe)
    pp_input_text = table.concat(tmpTokens, ' ')
    print('BPE-ed input text: ' .. pp_input_text)
  else
    pp_input_text, num_tab = digit_placeholder_prep(input_text)
  end
  

  -- tokenize & feature extraction
  local srcTokens = {}
  for word in pp_input_text:gmatch'([^%s]+)' do
    table.insert(srcTokens, word)
  end

  table.insert(batch, translator:buildInput(srcTokens))

  -- 170302. translator:buildData Added, to access
  pla = require 'pl.pretty'
  local unk_word_cnt = 0
  local data = translator:buildData(batch) 

  if is_mixed_param then
    pp_input_text = ''
  end
  --print (pla.dump(data))
  local idxed_words = data.src[1]
  for i = 1, idxed_words:storage():size() do
    if idxed_words[i] == onmt.Constants.UNK then
      print('unk found.' .. srcTokens[i])
      if is_mixed_param then
	print(srcTokens[i])
        local new_word = break_oov_into_chars(srcTokens[i])
	if i > 1 then
          pp_input_text = pp_input_text .. ' '
        end  
	pp_input_text = pp_input_text .. new_word
      else
        unk_word_cnt = unk_word_cnt + 1
      end
    else
      if is_mixed_param then
	if i > 1 then
          pp_input_text = pp_input_text .. ' '
        end  
        pp_input_text = pp_input_text .. srcTokens[i]
      end
    end 
  end
  print('unk word count: ' .. unk_word_cnt)

  if is_mixed_param and unk_word_cnt >= 1 then
    print('mixed - rebuild input start.')
    srcTokens = {}
    batch = {}
    for word in pp_input_text:gmatch'([^%s]+)' do
      table.insert(srcTokens, word)
    end
    print ('-mixed: ' .. pp_input_text)

    table.insert(batch, translator:buildInput(srcTokens))
    local data = translator:buildData(batch) 
    unk_word_cnt = 0
    local idxed_words = data.src[1]
    for i = 1, idxed_words:storage():size() do
      if idxed_words[i] == onmt.Constants.UNK then
        unk_word_cnt = unk_word_cnt + 1
      end
    end
  end

  -- Translate
  local results = translator:translate(batch)

  -- result = predicted output text
  local pred_result = translator:buildOutput(results[1].preds[1])
  local pred_score = results[1].preds[1].score

  print ('predSent: ' .. pred_result)
  print ('score: ' .. tostring(pred_score))

  -- pred_result, pred_score, nbests = beam.search(input_text)
  -- FIXME: Prediction PPL을 구해야 하는 경우, math.exp(-pred_score/#pred_result-1)로 계산 가능하다. totalScore를 그것으로 반환해야 하는지는
  -- 고민이 필요함.
  _, word_token_len = pred_result:gsub("%S+", "")
  if (DEBUG_PRINT == true) then
    print('input:', input_text)
    print('result:', pred_result, 'pred-score:', pred_score, 'normalized perplexity:', math.exp(pred_score/word_token_len))
  end

  max_num_idx = num_idx
  recov_num_idx = 1
  -- unk penalty를 가한다
  pred_score = pred_score - (unk_word_cnt * unk_penalty) 
  normed_score = math.exp(pred_score/word_token_len)

  -- 문두 대문자 처리
  pred_result = pred_result:sub(1, 1):upper() .. pred_result:sub(2)

  while true do
	if id_digit_align_enable == true then
		-- 원래 __digit_[0-9]+ 가 되어야 하나, 영어쪽 corpus 수정 실패로 인해 임시로 수정. 10개는 넘어가지 않을 것이다.
		--sidx, eidx = pred_result:find("__digit_[0-9]__") 	-- enable when ko-en testing 365m
		sidx, eidx = pred_result:find("__digit_[0-9]")
		if sidx == nil then break end
		--an_id = pred_result:sub(sidx+8, eidx-2)		-- enable with ko-en testing 365m
		an_id = pred_result:sub(sidx+8, eidx)
		-- 숫자 다음에 0이 오면 공백을 제거
		print (#pred_result)
		if eidx+2 < #pred_result then
		    local next_wrd = pred_result:sub(eidx+1, eidx+2)
		    --print ('next_wrd: [' .. next_wrd .. ']')
		    if next_wrd == " 0" then
  	 	        pred_result = pred_result:sub(1, eidx) .. pred_result:sub(eidx+2)
			--print ('new res: ' .. pred_result)
		    end
		end
		if num_tab[tonumber(an_id)] == nil then
			--pred_result = ""
			pred_score = -150.0
			normed_score = 0.00000001
			break
		else
			pred_result = pred_result:sub(1, sidx-1) .. num_tab[tonumber(an_id)] .. pred_result:sub(eidx + 1)
		end
	else
		sidx, eidx = pred_result:find("__digit_")
		if sidx == nil then break end
		pred_result = pred_result:gsub("__digit_", num_tab[recov_num_idx], 1)
	end
	recov_num_idx = recov_num_idx + 1
  end

  if is_mixed then
    pred_result = pred_result:gsub('<B>', '')
    pred_result = pred_result:gsub(' <M>', '')
    pred_result = pred_result:gsub(' <E>', '')
    pred_result = pred_result:gsub(' ￭', '')
  end

  if _G.bpe ~= nil then
    pred_result = pred_result:gsub('￭ ', '')
    pred_result = pred_result:gsub(' ￭', '')
  end

  -- unk penalty를 가함 --
  local usidx, ueidx = pred_result:find("<unk>")
  if usidx ~= nil then
    pred_score = -150.0
    normed_score = 0.00000001
  end

  return pred_result, pred_score, normed_score 
end

local url_matching_rules = { 
  {
    match = "^/RPC2/?$",
    with = xavante_wsapi_inst.makeHandler(wsapi_handler)
  }
}

local function main()
  print('XML-RPC Server for OpenNMT, by ETRI Language Intelligence Research Group. 2016.')

  local opt = cmd:parse(arg)

  is_mixed = opt.mixed

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  _G.logger:info("Initializing NMT Model, Please wait for a while...")

  if is_mixed then
    _G.logger:info('Mixed Word/Character Mode enabled.')
  end

  -- BPE를 적용한다.
  if #opt.bpe_model > 0 then
    _G.BPE = require ('tools.utils.BPE')
    _G.tokenizer = require('tools.utils.tokenizer')
    _G.bpe = _G.BPE.new(opt.bpe_model, true, false)
    _G.logger:info('BPE model enabled: joiner_annotate - true mode')
  end

  -- XML-RPC Handler에서 접근하기 위해, translator는 global로 잡는다
  translator = onmt.translate.Translator.new(opt)

  local server_config = {
    server = {
      host = opt.host,
      port = opt.port 
    },
    defaultHost = {
      rules = url_matching_rules
    }
  }
  _G.logger:info('Initialize Complete, Now register XML-RPC methods & Start XML-RPC HTTPD Server.')
  xmlrpc.srvMethods(xmlrpc_exports)
  xavante.HTTP(server_config)
  xavante.start()
end

main()
