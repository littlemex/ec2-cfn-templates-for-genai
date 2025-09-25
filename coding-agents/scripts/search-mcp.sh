#!/bin/bash
# Web Search MCP Server Wrapper - 完全版
# すべてのログメッセージを除去し、JSONレスポンスのみを通す

exec node /work/mcp-servers/web-search-mcp/dist/index.js 2>/dev/null | (
    while IFS= read -r line; do
        # JSONかどうかを厳密にチェック
        if echo "$line" | jq . >/dev/null 2>&1; then
            # 有効なJSONのみを出力
            echo "$line"
        fi
        # 非JSONメッセージは完全に無視
    done
)
