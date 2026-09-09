#!/bin/bash
input=$(cat)
model=$(echo "$input" | jq -r '.model.display_name // empty' 2>/dev/null)
model_id=$(echo "$input" | jq -r '.model.id // empty' 2>/dev/null)
used=$(echo "$input" | jq -r '.context_window.used_percentage // empty' 2>/dev/null)
ctx_bar=""; ctx_pct=""
if [ -n "$used" ] && [ "$used" != "null" ]; then
  ctx_pct=${used%.*}
  [ -z "$ctx_pct" ] && ctx_pct=0
  fill=$(awk -v p="$used" 'BEGIN{printf "%d", (p/10+0.5)}')
  [ "$fill" -gt 10 ] && fill=10; [ "$fill" -lt 0 ] && fill=0
  bar=""; i=1; while [ $i -le 10 ]; do if [ $i -le $fill ]; then bar="${bar}█"; else bar="${bar}░"; fi; i=$((i+1)); done
  if [ "${ctx_pct:-0}" -gt 90 ]; then c=31; elif [ "${ctx_pct:-0}" -gt 70 ]; then c=33; else c=32; fi
  ctx_bar=$(printf '\033[%sm%s\033[0m' "$c" "$bar")
  ctx_pct=$(printf '%.0f' "$used")
else
  bar="░░░░░░░░░░"; ctx_bar="$bar"; ctx_pct="0"
  c=90
fi
cache="/tmp/opencode-go-usage.json"
ttl=60
now=$(date +%s)
js=""
if [ -f "$cache" ]; then mt=$(stat -f %m "$cache" 2>/dev/null || stat -c %Y "$cache" 2>/dev/null); age=$((now-mt)); [ "$age" -lt "$ttl" ] && js=$(cat "$cache" 2>/dev/null); fi
if [ -z "$js" ]; then
  tok=$(jq -r '.["opencode-go"].key // empty' ~/.local/share/opencode/auth.json 2>/dev/null)
  [ -z "$tok" ] && tok=$(jq -r '.Providers[] | select(.name=="opencode") | .api_key // empty' ~/.claude-code-router/config.json 2>/dev/null)
  if [ -n "$tok" ]; then
    resp=$(curl -s --max-time 2 -H "Authorization: Bearer $tok" -H "User-Agent: opencode/2.1.131" https://opencode.ai/zen/go/v1/usage 2>/dev/null)
    if echo "$resp" | jq -e '.usage' >/dev/null 2>&1; then echo "$resp" > "$cache" 2>/dev/null; js="$resp"; else [ -f "$cache" ] && js=$(cat "$cache" 2>/dev/null); fi
  fi
fi
go_bar="░░░░░░░░░░"; go_pct="0"; gc=90
if [ -n "$js" ]; then
  w=$(echo "$js" | jq -r '.usage.weekly.percent // empty' 2>/dev/null)
  m=$(echo "$js" | jq -r '.usage.monthly.percent // empty' 2>/dev/null)
  r=$(echo "$js" | jq -r '.usage.rolling.percent // empty' 2>/dev/null)
  pct="${m:-$w}"; pct="${pct:-$r}"
  if [ -n "$pct" ] && [ "$pct" != "null" ]; then
    go_pct=$(printf '%.0f' "$pct")
    p_int=${pct%.*}
    fill=$(awk -v p="$pct" 'BEGIN{printf "%d", (p/10+0.5)}')
    [ "$fill" -gt 10 ] && fill=10; [ "$fill" -lt 0 ] && fill=0
    bar=""; i=1; while [ $i -le 10 ]; do if [ $i -le $fill ]; then bar="${bar}█"; else bar="${bar}░"; fi; i=$((i+1)); done
    if [ "${p_int:-0}" -ge 90 ]; then gc=31; elif [ "${p_int:-0}" -ge 70 ]; then gc=33; else gc=32; fi
    go_bar=$(printf '\033[%sm%s\033[0m' "$gc" "$bar")
  fi
fi
[ -z "$model" ] || [ "$model" = "null" ] && model="Unknown"
# Antigravity quota bar (only when current model is antigravity)
anti_info=""
model_lc=$(printf '%s %s' "$model" "$model_id" | tr '[:upper:]' '[:lower:]')
case "$model_lc" in
  *antigravity*)
    ag_cache="/tmp/antigravity-models.json"
    ag_ttl=300
    ag_js=""
    if [ -f "$ag_cache" ]; then ag_mt=$(stat -f %m "$ag_cache" 2>/dev/null || stat -c %Y "$ag_cache" 2>/dev/null); ag_age=$((now-ag_mt)); [ "$ag_age" -lt "$ag_ttl" ] && ag_js=$(cat "$ag_cache" 2>/dev/null); fi
    if [ -z "$ag_js" ]; then
      ag_tok=$(jq -r '.activeEmail as $a | .accounts[] | select(.email==$a) | .access_token // empty' ~/.claude-code-router/antigravity-auth.json 2>/dev/null)
      if [ -n "$ag_tok" ]; then
        ag_resp=$(curl -s --max-time 2 -H "Authorization: Bearer $ag_tok" -H "Content-Type: application/json" -H "User-Agent: antigravity/1.23.2 DARWIN_ARM64" https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels -d '{}' 2>/dev/null)
        if echo "$ag_resp" | jq -e '.models' >/dev/null 2>&1; then echo "$ag_resp" > "$ag_cache" 2>/dev/null; ag_js="$ag_resp"; else [ -f "$ag_cache" ] && ag_js=$(cat "$ag_cache" 2>/dev/null); fi
      fi
    fi
    ag_frac=""
    # Preferred: true depletion from the local IDE language server
    # (planStatus available vs monthly prompt credits). The per-model
    # remainingFraction below stays pinned at 1.0 on unlimited-style
    # plans, so it cannot show real usage. Discovery (pid -> ports ->
    # csrf token) runs at most once per TTL; cached value is reused.
    ag_credits_cache="/tmp/antigravity-credits.json"
    ag_credits_ttl=120
    ag_credits_json=""
    if [ -f "$ag_credits_cache" ]; then
      cx_mt=$(stat -f %m "$ag_credits_cache" 2>/dev/null || stat -c %Y "$ag_credits_cache" 2>/dev/null)
      cx_age=$((now-cx_mt))
      [ "$cx_age" -lt "$ag_credits_ttl" ] && ag_credits_json=$(cat "$ag_credits_cache" 2>/dev/null)
    fi
    if [ -z "$ag_credits_json" ]; then
      for ag_pid in $(pgrep -f "language_server_macos_arm" 2>/dev/null); do
        ag_tok=$(ps -p "$ag_pid" -o args= 2>/dev/null | grep -o '\-\-csrf_token [a-f0-9-]*' | awk '{print $2}' | head -n 1)
        [ -z "$ag_tok" ] && continue
        for ag_port in $(lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | awk -v pid="$ag_pid" '$2==pid {print $9}' | grep -o '[0-9]*$' | sort -u); do
          ag_status=$(curl -sk --max-time 3 -X POST "https://127.0.0.1:$ag_port/exa.language_server_pb.LanguageServerService/GetUserStatus" -H "Content-Type: application/json" -H "Connect-Protocol-Version: 1" -H "X-Codeium-Csrf-Token: $ag_tok" -d '{"metadata":{"ideName":"antigravity","extensionName":"antigravity","locale":"en"}}' 2>/dev/null)
          ag_avail=$(printf '%s' "$ag_status" | jq -r '.userStatus.planStatus.availablePromptCredits // empty' 2>/dev/null)
          ag_monthly=$(printf '%s' "$ag_status" | jq -r '.userStatus.planStatus.planInfo.monthlyPromptCredits // empty' 2>/dev/null)
          if [ -n "$ag_avail" ] && [ -n "$ag_monthly" ] && [ "$ag_avail" != "null" ] && [ "$ag_monthly" != "null" ] && [ "$ag_monthly" != "0" ]; then
            ag_credits_json=$(printf '{"available":%s,"monthly":%s}' "$ag_avail" "$ag_monthly")
            printf '%s' "$ag_credits_json" > "$ag_credits_cache" 2>/dev/null
            break 2
          fi
        done
      done
    fi
    if [ -n "$ag_credits_json" ]; then
      ag_cpct=$(printf '%s' "$ag_credits_json" | jq -r '.available as $a | .monthly as $m | if ($m | type) == "number" and $m > 0 and (($a | type) == "number") then (($a/$m)*100) else empty end' 2>/dev/null)
      if [ -n "$ag_cpct" ] && [ "$ag_cpct" != "null" ]; then
        ag_frac=$(awk -v p="$ag_cpct" 'BEGIN{printf "%.4f", p/100}')
      fi
    fi
    if [ -z "$ag_frac" ] && [ -n "$ag_js" ]; then
      ag_src="$model"
      case "$(printf '%s' "$model_id" | tr '[:upper:]' '[:lower:]')" in *antigravity*) ag_src="$model_id";; esac
      ag_slug=$(printf '%s' "$ag_src" | sed 's|.*/||' | sed 's/\[.*//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]' | sed 's/ (.*//' | sed 's/[ _]/-/g')
      ag_frac=""
      for cand in "$ag_slug" "$(printf '%s' "$ag_slug" | tr '.' '-')" "$(printf '%s' "$ag_slug" | sed 's/-preview$//')" "$(printf '%s' "$ag_slug" | tr '.' '-' | sed 's/-preview$//')"; do
        for c2 in "$cand" "$(printf '%s' "$cand" | sed -E 's/-(high|medium|low)$/-tiered/')"; do
          ag_frac=$(printf '%s' "$ag_js" | jq -r --arg m "$c2" '.models[$m].quotaInfo.remainingFraction // empty' 2>/dev/null)
          [ -n "$ag_frac" ] && [ "$ag_frac" != "null" ] && break 2
        done
      done
    fi
    if [ -n "$ag_frac" ] && [ "$ag_frac" != "null" ]; then
      ag_pct=$(awk -v f="$ag_frac" 'BEGIN{printf "%.0f", f*100}')
      ag_fill=$(awk -v f="$ag_frac" 'BEGIN{printf "%d", (f*10+0.5)}')
      [ "$ag_fill" -gt 10 ] && ag_fill=10; [ "$ag_fill" -lt 0 ] && ag_fill=0
      ag_bar=""; i=1; while [ $i -le 10 ]; do if [ $i -le $ag_fill ]; then ag_bar="${ag_bar}█"; else ag_bar="${ag_bar}░"; fi; i=$((i+1)); done
      if [ "${ag_pct:-0}" -lt 10 ]; then ac=31; elif [ "${ag_pct:-0}" -lt 30 ]; then ac=33; else ac=32; fi
      ag_bar_c=$(printf '\033[%sm%s\033[0m' "$ac" "$ag_bar")
      anti_info=$(printf ' | anti::%s %s%%' "$ag_bar_c" "$ag_pct")
    fi
    ;;
esac
# Codex usage bar (only when current model is codex-provider).
# Usage data is written to the cache by the gateway codex transformer
# (plain script curl cannot pass the Cloudflare check upstream).
codex_info=""
cx_show=""
case "$model_lc" in
  *codex/*) cx_show=1 ;;
  */*) ;; # other provider-qualified model -> not codex
  *gpt-reserve*|*gpt-5.6-terra*|*gpt-5.6-luna*|*gpt-5.5*|*gpt-5.4-mini*|*codex-auto-review*) cx_show=1 ;;
esac
if [ -n "$cx_show" ]; then
  cx_js=$(cat "/tmp/codex-usage.json" 2>/dev/null)
  if [ -n "$cx_js" ]; then
    cx_pct=$(echo "$cx_js" | jq -r '.used_percent // empty' 2>/dev/null)
    cx_reached=$(echo "$cx_js" | jq -r '.limit_reached // false' 2>/dev/null)
    if [ -n "$cx_pct" ] && [ "$cx_pct" != "null" ]; then
      cx_pct_i=$(printf '%.0f' "$cx_pct")
      cx_fill=$(awk -v p="$cx_pct" 'BEGIN{printf "%d", (p/10+0.5)}')
      [ "$cx_fill" -gt 10 ] && cx_fill=10; [ "$cx_fill" -lt 0 ] && cx_fill=0
      cx_bar=""; i=1; while [ $i -le 10 ]; do if [ $i -le $cx_fill ]; then cx_bar="${cx_bar}█"; else cx_bar="${cx_bar}░"; fi; i=$((i+1)); done
      if [ "$cx_reached" = "true" ] || [ "${cx_pct_i:-0}" -ge 90 ]; then cc=31; elif [ "${cx_pct_i:-0}" -ge 70 ]; then cc=33; else cc=32; fi
      cx_bar_c=$(printf '\033[%sm%s\033[0m' "$cc" "$cx_bar")
      codex_info=$(printf ' | codex::%s %s%%' "$cx_bar_c" "$cx_pct_i")
    fi
  fi
fi
# CCR latest log: status, model, duration
ccr_info=""
if command -v sqlite3 >/dev/null 2>&1; then
  row=$(sqlite3 ~/.claude-code-router/app-data/request-logs.sqlite "SELECT status_code, coalesce(resolved_model, requested_model, model), duration_ms FROM request_logs ORDER BY id DESC LIMIT 1;" 2>/dev/null)
  if [ -n "$row" ]; then
    IFS='|' read -r ccr_status ccr_model ccr_dur <<< "$row"
    ccr_model=$(basename "$ccr_model" 2>/dev/null | sed 's/::.*//')
    [ -z "$ccr_model" ] && ccr_model="unknown"
    # duration formatting
    if [ -n "$ccr_dur" ] && [ "$ccr_dur" != "0" ]; then
      if [ "$ccr_dur" -ge 1000 ]; then
        ccr_dur_s=$(awk -v d="$ccr_dur" 'BEGIN{printf "%.1fs", d/1000}')
      else
        ccr_dur_s="${ccr_dur}ms"
      fi
    else
      ccr_dur_s="0ms"
    fi
    # status color
    if [ "$ccr_status" -ge 200 ] && [ "$ccr_status" -lt 300 ]; then sc=32; elif [ "$ccr_status" -ge 400 ]; then sc=31; else sc=33; fi
    ccr_status_c=$(printf '\033[%sm%s\033[0m' "$sc" "$ccr_status")
    ccr_info=$(printf ' | ccr::%s %s %s' "$ccr_status_c" "$ccr_model" "$ccr_dur_s")
  fi
fi
printf 'context::%s %s%% | go-usage::%s %s%% | %s%s%s%s\n' "$ctx_bar" "$ctx_pct" "$go_bar" "$go_pct" "$model" "$anti_info" "$codex_info" "$ccr_info"
