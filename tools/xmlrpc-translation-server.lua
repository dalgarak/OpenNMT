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

local cmd = torch.CmdLine()

-- translator를 밖으로 빼놓음
local translator

cmd:text("")
cmd:text("**onmt.xmlrpc-translation_server**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])
onmt.translate.Translator.declareOpts(cmd)

cmd:option('-host', '*', [[Host to run the server on]])
cmd:option('-port', '8020', [[Port to run the server on]])
cmd:option('-extra_digit', 'false', [[If it enabled, comma between numerics will be omitted.]])
cmd:text("")
cmd:text("**Other options**")
cmd:text("")
cmd:option('-gpuid', -1, [[ID of the GPU to use (-1 = use CPU, 0 = let cuda choose between available GPUs)]])
cmd:option('-fallback_to_cpu', false, [[If = true, fallback to CPU if no GPU available]])

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

-- XML-RPC WSAPI Handler
function wsapi_handler(wsapi_env)
  local headers = { ["Content-type"] = "text/xml" }
  local req = wsapi_request.new(wsapi_env)
  local method, arg_table = xmlrpc.srvDecode(req.POST.post_data)
  --[[ arg_table의 경우, { 1: { "text" = value, "..." = "..." } } 와 같이
      들어오기 때문에, [1]로 한번 unpacking을 해 줘야 한다.
  ]]
  local func = xmlrpc.dispatch(method)
  local result = { pcall(func, arg_table[1]["text"]) }
  --print ('result table: ', result)
  local is_ok = result[1]	-- pcall의 처리 결과가 나온다
  local final_result = {}

  if not is_ok then
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

-- translate method
--[[ 
      Each XML-RPC procedures returns pcall result(true/false), return of procedures 1, return of procedures 2 ... as a lua table.
  ]]
function xmlrpc_exports.translate(input_text)
  local unk_penalty = 2 
  local batch = {}

  -- Input text의 숫자를 읽어들여, 이를 placeholder로 바꿔주도록 한다.
  local pp_input_text, num_tab = digit_placeholder_prep(input_text)

  -- tokenize & feature extraction
  local srcTokens = {}
  for word in pp_input_text:gmatch'([^%s]+)' do
    table.insert(srcTokens, word)
  end

  table.insert(batch, translator:buildInput(srcTokens))

  -- data.src에 idx화 된 데이터가 들어 있다
  pla = require 'pl.pretty'
  local unk_word_cnt = 0
  dsrc = data.words[1]:storage()
  --print (pla.dump(dsrc))
  for i = 1, dsrc:size() do
    if dsrc[i] == onmt.Constants.UNK then
      unk_word_cnt = unk_word_cnt + 1
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
		sidx, eidx = pred_result:find("__digit_[0-9]")
		if sidx == nil then break end
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
			pred_result = ""
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
  local requiredOptions = {
    "model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  _G.logger:info("Initializing NMT Model, Please wait for a while...")

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
