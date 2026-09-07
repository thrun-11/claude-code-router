#!/bin/bash
input=$(cat)
model=$(echo "$input" | jq -r '.model.display_name // empty' 2>/dev/null)
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
printf 'context::%s %s%% | go-usage::%s %s%% | %s%s\n' "$ctx_bar" "$ctx_pct" "$go_bar" "$go_pct" "$model" "$ccr_info"
