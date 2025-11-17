# Historical Data Backfill

## Overview

This backfill system progressively retrieves 30 days of historical job data by scraping 2-day windows, designed to run hourly via GitHub Actions to avoid long-running scripts.

## How It Works

1. **15 Windows**: Divides 30 days into fifteen 2-day windows
2. **Hourly Execution**: GitHub Action runs every hour, processing one window per run
3. **State Tracking**: Progress saved in `data/backfill_state.json` to resume across runs
4. **Auto-Completion**: Automatically stops when all 15 windows are complete (~15 hours total)

### Time Windows (Working Backwards)

- Window 0: Days 30-28 ago (oldest)
- Window 1: Days 28-26 ago
- ...
- Window 14: Days 2-0 ago (most recent)

## Setup

### 1. Local Testing (Optional)

Test the script locally before deploying:

```powershell
# Install dependencies
pip install -r requirements-actions.txt

# Run one window
python scripts/backfill_historical.py

# Check progress
cat data/backfill_state.json

# Reset and start over (if needed)
python scripts/backfill_historical.py --reset
```

### 2. GitHub Actions Setup

The workflow is already configured in `.github/workflows/backfill_historical.yml`

**Required Secrets** (should already be set):
- `S3_BUCKET_NAME`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

**Optional Secrets**:
- `S3_RAW_KEY` (default: raw_jobs.csv)
- `S3_PROCESSED_KEY` (default: processed_jobs.csv)

### 3. Start Backfill

**Option A: Automatic (Hourly)**
- The workflow runs automatically every hour once pushed to GitHub
- No action needed - just wait 15 hours for completion

**Option B: Manual Trigger**
1. Go to GitHub repo → Actions tab
2. Select "Historical Data Backfill (Hourly)"
3. Click "Run workflow"
4. Repeat hourly or wait for automatic runs

## Monitoring Progress

### Check Logs
- Go to Actions tab → Click on latest "Historical Data Backfill" run
- View logs to see current window and progress

### Expected Output
```
==============================================================
📊 Backfill Progress: Window 5/15
📅 Time Range: Days 20-18 ago
==============================================================

✓ Loaded existing raw from S3: 45,231 rows
🔍 Scraping window 4: Days 20-18 ago
✓ Scraped 1,847 jobs from this window
📝 Deduplication: removed 23 by ID, 41 by composite key
   Total: 47,014 rows (added 1,783 net new)
✓ Uploaded raw to S3
✓ Uploaded processed to S3

✅ Window 5/15 complete!
📊 Progress: 5/15 windows done
⏳ Remaining: 10 windows (~10 hours at 1 window/hour)
```

### Completion Status
When complete, you'll see:
```
✅ Backfill complete! All 15 windows (30 days) have been processed.
```

## Customization

Edit `.github/workflows/backfill_historical.yml` to change:

```yaml
env:
  SEARCH_TERMS: "Data Scientist,ML Engineer"  # Change search terms
  LOCATIONS: "San Francisco,New York"          # Change locations
  RESULTS_PER_SITE: "400"                      # Adjust results per window
```

## Stopping the Backfill

### Temporary Pause
Disable the workflow in GitHub:
1. Actions → Historical Data Backfill → "..." menu
2. Click "Disable workflow"

### Resume Later
Re-enable the workflow - it will continue from where it left off using saved state

### Permanent Stop
Delete `.github/workflows/backfill_historical.yml`

## Resetting and Starting Over

### Local Reset
```powershell
python scripts/backfill_historical.py --reset
```

### GitHub Reset
1. Go to Actions → Latest run
2. Download "backfill-state" artifact
3. Edit `backfill_state.json` to reset:
```json
{
  "completed_windows": [],
  "current_window": 0,
  "total_windows": 15
}
```
4. Manually trigger next run

## Data Management

### Deduplication
The script automatically:
- Merges new data with existing S3 data
- Removes duplicates by job ID
- Removes duplicates by normalized (title + company + location + link)

### Storage
- **Local**: `data/raw_jobs.csv` and `data/processed_jobs.csv`
- **S3**: Uploaded after each successful window
- **State**: `data/backfill_state.json` (preserved across GitHub Action runs)

## Troubleshooting

### Script Fails Mid-Window
- State is preserved - next run will retry the same window
- Check logs for specific error (rate limiting, network issues, etc.)

### Duplicate Data
- Script handles deduplication automatically
- If concerned, can manually reset and restart

### Want Faster Completion
- **Not recommended**: Running multiple windows in parallel may cause rate limiting
- Better: Wait for hourly runs or trigger manually more frequently

### State File Issues
If state becomes corrupted:
1. Delete artifact in GitHub Actions
2. Script will start fresh from Window 0

## Integration with Daily Refresh

**Recommended Setup**:
1. Run this backfill once to get historical data (15 hours)
2. Keep `refresh_s3_24h.yml` running daily for ongoing updates
3. Disable backfill workflow after completion to avoid re-scraping old data

## Performance Notes

- Each window scrapes ~400 results/site × 2 sites × 2 locations × 5 search terms = ~8,000 jobs max
- Actual results vary by availability
- Deduplication typically reduces final count by 20-40%
- Full 30-day backfill: ~30,000-50,000 unique jobs (varies by market)

## Questions?

See main README.md or check the scripts' inline documentation:
- `scripts/backfill_historical.py` - Main backfill logic
- `.github/workflows/backfill_historical.yml` - GitHub Action configuration
