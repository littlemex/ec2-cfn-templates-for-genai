#!/bin/bash
# setup_perfetto_mcp.sh
# ======================
#
# AWS Neuron Hardware Deep Analysis結果をPerfetto MCPで詳細解析するための
# 自動設定スクリプト（shellスクリプト版）
#
# Usage: bash setup_perfetto_mcp.sh
#        sudo bash setup_perfetto_mcp.sh  (if needed)

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎨 Perfetto MCP Setup for AWS Neuron Hardware Analysis${NC}"
echo "============================================================"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Verify local perfetto-mcp installation
echo
echo "🔧 Verifying local perfetto-mcp installation..."

PERFETTO_MCP_DIR="/work/pytorch-neuron/perfetto-mcp"

if [[ -d "$PERFETTO_MCP_DIR" && -f "$PERFETTO_MCP_DIR/src/perfetto_mcp/__init__.py" ]]; then
    print_info "Local perfetto-mcp found at: $PERFETTO_MCP_DIR"
    
    # Test local installation
    echo "Testing local perfetto-mcp installation..."
    if cd "$PERFETTO_MCP_DIR" && uv run -m perfetto_mcp --help &>/dev/null; then
        print_info "Local perfetto-mcp is working correctly"
    else
        print_warning "Local perfetto-mcp test failed, but proceeding with setup"
    fi
    cd - > /dev/null
else
    print_error "Local perfetto-mcp not found at: $PERFETTO_MCP_DIR"
    echo "Please ensure the perfetto-mcp directory exists with source code"
    exit 1
fi

# Step 2: Detect Cline MCP configuration path
echo
echo "📁 Detecting Cline MCP configuration paths..."

# Use coder user for MCP settings (regardless of who runs the script)
CURRENT_USER="coder"
echo "Target user for MCP settings: $CURRENT_USER"

# Define Cline MCP settings path (always use coder user)
CLINE_MCP_DIR="/home/$CURRENT_USER/.local/share/code-server/User/globalStorage/saoudrizwan.claude-dev/settings"
CLINE_MCP_FILE="$CLINE_MCP_DIR/cline_mcp_settings.json"

# Alternative Amazon Q path
AMAZONQ_MCP_DIR="/home/$CURRENT_USER/.aws/amazonq"
AMAZONQ_MCP_FILE="$AMAZONQ_MCP_DIR/mcp.json"

echo "Cline MCP settings path: $CLINE_MCP_FILE"
echo "Amazon Q MCP settings path: $AMAZONQ_MCP_FILE"

# Step 3: Create directories if they don't exist
echo
echo "📝 Creating MCP configuration directories..."

mkdir -p "$CLINE_MCP_DIR"
mkdir -p "$AMAZONQ_MCP_DIR"

print_info "MCP directories created/verified"

# Step 4: Update Cline MCP configuration
echo
echo "🔧 Updating Cline MCP configuration..."

# Check if jq is available
if ! command -v jq &> /dev/null; then
    print_warning "jq not found. Installing jq..."
    sudo apt-get update -qq && sudo apt-get install -y jq
fi

# Create or update Cline MCP settings
if [[ -f "$CLINE_MCP_FILE" ]]; then
    print_info "Existing Cline MCP config found, updating..."
    
    # Backup existing config
    cp "$CLINE_MCP_FILE" "$CLINE_MCP_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Add perfetto-mcp-local to existing config (using local installation)
    jq '.mcpServers["perfetto-mcp-local"] = {
        "autoApprove": [
            "find_slices",
            "execute_sql_query"
        ],
        "disabled": false,
        "command": "uv",
        "args": [
            "--directory",
            "/work/pytorch-neuron/perfetto-mcp",
            "run",
            "-m",
            "perfetto_mcp"
        ],
        "env": {
            "PYTHONPATH": "src"
        },
        "transportType": "stdio"
    }' "$CLINE_MCP_FILE" > "$CLINE_MCP_FILE.tmp"
    
    mv "$CLINE_MCP_FILE.tmp" "$CLINE_MCP_FILE"
    
else
    print_info "Creating new Cline MCP config..."
    
    cat > "$CLINE_MCP_FILE" << 'EOF'
{
  "mcpServers": {
    "awslabs.aws-documentation-mcp-server": {
      "autoApprove": [
        "read_documentation",
        "search_documentation",
        "recommend"
      ],
      "disabled": false,
      "command": "uvx",
      "args": [
        "awslabs.aws-documentation-mcp-server@latest"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR"
      },
      "transportType": "stdio"
    },
    "perfetto-mcp-local": {
      "autoApprove": [
        "find_slices",
        "execute_sql_query"
      ],
      "disabled": false,
      "command": "/home/coder/.local/bin/uv",
      "args": [
        "--directory",
        "/work/pytorch-neuron/perfetto-mcp",
        "run",
        "-m",
        "perfetto_mcp"
      ],
      "env": {
        "PYTHONPATH": "src"
      },
      "transportType": "stdio"
    }
  }
}
EOF
fi

# Ensure proper ownership (keep coder user ownership even if run with sudo)
if [[ "$EUID" -eq 0 ]]; then
    # If running as root, ensure files stay owned by coder
    chown -R coder:coder "$CLINE_MCP_DIR" 2>/dev/null || true
    chown -R coder:coder "$AMAZONQ_MCP_DIR" 2>/dev/null || true
fi

print_info "Cline MCP configuration updated successfully"

# Step 5: Find and verify Perfetto files
echo
echo "🔍 Searching for generated Perfetto files..."

PERFETTO_FILES=()

# Search in known directories
SEARCH_DIRS=(
    "/tmp/neuron_hardware_profiles_comprehensive_hardware_deep_analysis"
    "/tmp"
    "."
)

for search_dir in "${SEARCH_DIRS[@]}"; do
    if [[ -d "$search_dir" ]]; then
        while IFS= read -r -d '' file; do
            PERFETTO_FILES+=("$file")
        done < <(find "$search_dir" -name "*.pftrace" -type f -print0 2>/dev/null)
    fi
done

# Remove duplicates
IFS=" " read -r -a UNIQUE_FILES <<< "$(printf '%s\n' "${PERFETTO_FILES[@]}" | sort -u | tr '\n' ' ')"

echo "📊 Found ${#UNIQUE_FILES[@]} Perfetto trace files:"
for pf in "${UNIQUE_FILES[@]}"; do
    if [[ -f "$pf" ]]; then
        file_size=$(stat -c%s "$pf" 2>/dev/null || echo "unknown")
        echo "   • $pf ($file_size bytes)"
    fi
done

# Step 6: Generate usage guide
echo
echo "📚 Generating usage guide..."

GUIDE_FILE="pytorch-neuron/docs/PERFETTO_MCP_USAGE_GUIDE.md"

cat > "$GUIDE_FILE" << EOF
# Perfetto MCP Setup Complete!

## ✅ Setup Summary
- Cline MCP config: $CLINE_MCP_FILE
- Local perfetto-mcp: /work/pytorch-neuron/perfetto-mcp
- ${#UNIQUE_FILES[@]} Perfetto trace files detected

## 🚀 Usage Instructions

### 1. Restart Cline/Code-server
Restart your Cline environment to load the new MCP configuration.

### 2. Verify MCP Connection
In Cline, check that perfetto-mcp-local is available:
\`\`\`
List available MCP tools
\`\`\`

### 3. Start Analysis
Use natural language queries with Perfetto MCP:

For vmap vs scan hardware analysis:
\`\`\`
Use perfetto trace ${UNIQUE_FILES[0]:-'/path/to/trace.pftrace'} for process analysis:

"Compare vmap vs scan hardware utilization patterns in neuron execution"

"Analyze for-loop size scaling performance differences"  

"Find memory access bottlenecks in execution timeline"

"Show compute engine utilization overlap between patterns"
\`\`\`

## 📊 Advanced Analysis Queries

### Hardware Comparison Analysis
\`\`\`
"Compare tensor engine utilization between vmap and scan patterns"
"Identify DMA transfer differences in for-loop vs vmap execution"
"Find memory bandwidth bottlenecks across all execution patterns"
\`\`\`

### Performance Bottleneck Detection  
\`\`\`
"Detect performance bottlenecks in neuron hardware timeline"
"Analyze memory-bound vs compute-bound execution characteristics"
"Find instruction dependency chains causing performance issues"
\`\`\`

### Timeline Deep Dive
\`\`\`
"Show detailed timeline of hardware events during vmap execution"
"Analyze sequential vs parallel hardware resource utilization"
"Identify engine overlap efficiency opportunities"
\`\`\`

## 📁 Available Perfetto Files
EOF

for pf in "${UNIQUE_FILES[@]}"; do
    echo "- $pf" >> "$GUIDE_FILE"
done

cat >> "$GUIDE_FILE" << 'EOF'

## 🔧 Troubleshooting

If MCP connection fails:
1. Restart Cline/Code-server completely
2. Check MCP config syntax: `jq . /path/to/cline_mcp_settings.json`
3. Verify local perfetto-mcp: `cd /work/pytorch-neuron/perfetto-mcp && uv run -m perfetto_mcp --help`
4. Check file permissions on MCP config directory

## 📝 Example Commands

### Basic Trace Analysis
```
Use perfetto trace /tmp/neuron_hardware_profiles_comprehensive_hardware_deep_analysis/perfetto_analysis_0.pftrace for analysis:

"Summarize the hardware execution timeline"
```

### Pattern Comparison
```
"Compare execution patterns between different trace files and identify performance differences"
```

Happy analyzing! 🚀
EOF

print_info "Usage guide saved: $GUIDE_FILE"

# Step 7: Generate analysis script
echo
echo "📋 Creating Perfetto analysis helper script..."

ANALYSIS_SCRIPT="pytorch-neuron/scripts/run_perfetto_analysis.sh"

cat > "$ANALYSIS_SCRIPT" << 'EOF'
#!/bin/bash
# run_perfetto_analysis.sh
# Perfetto MCP Analysis Helper Script

echo "🎨 Perfetto MCP Analysis Helper"
echo "==============================="
echo
echo "Available Perfetto files:"
find /tmp -name "*.pftrace" -type f 2>/dev/null | head -10

echo
echo "🚀 Example Cline queries:"
echo
echo "1. Basic analysis:"
echo '   "Use perfetto trace /path/to/file.pftrace for neuron hardware analysis"'
echo
echo "2. Pattern comparison:"
echo '   "Compare vmap vs scan hardware utilization patterns"'
echo  
echo "3. Performance bottleneck detection:"
echo '   "Find performance bottlenecks in neuron hardware timeline"'
echo
echo "📚 See $PWD/docs/PERFETTO_MCP_USAGE_GUIDE.md for detailed usage instructions"
EOF

chmod +x "$ANALYSIS_SCRIPT"

print_info "Analysis helper script created: $ANALYSIS_SCRIPT"

# Step 8: Final verification
echo
echo "🔬 Final verification..."

# Check MCP config syntax
if jq empty "$CLINE_MCP_FILE" 2>/dev/null; then
    print_info "Cline MCP config JSON syntax is valid"
else
    print_error "Cline MCP config JSON syntax error"
    echo "Config file: $CLINE_MCP_FILE"
fi

# Check local perfetto-mcp installation
if cd "$PERFETTO_MCP_DIR" && uv run -m perfetto_mcp --help &>/dev/null; then
    print_info "Local perfetto-mcp is working correctly"
else
    print_warning "Local perfetto-mcp test failed"
fi
cd - > /dev/null

# Summary
echo
echo "🎉 Perfetto MCP setup completed successfully!"
echo
echo "📋 Summary:"
echo "   • Local perfetto-mcp: $PERFETTO_MCP_DIR"
echo "   • Cline MCP config: $CLINE_MCP_FILE" 
echo "   • Perfetto files found: ${#UNIQUE_FILES[@]}"
echo "   • Usage guide: $GUIDE_FILE"
echo "   • Helper script: $ANALYSIS_SCRIPT"
echo
echo "🔄 Next steps:"
echo "   1. Restart Cline/Code-server to load new MCP configuration"
echo "   2. Verify MCP connection in Cline with: 'List available MCP tools'"
echo "   3. Start analyzing with natural language queries!"
echo
echo "Example first query:"
echo '   "List available MCP tools"'
echo

exit 0
