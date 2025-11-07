#!/usr/bin/env python3
"""
Example usage of the targeted metrics exporter
"""
import os
import sys

# Set up the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def main():
    """Demonstrate usage of the targeted exporter"""
    print("🎯 Targeted Container Metrics Export")
    print("=" * 50)
    
    # Import the exporter class
    try:
        from export_metrics_targeted import ContainerMetricsExporter
    except ImportError:
        print("❌ Could not import ContainerMetricsExporter")
        print("Make sure export_metrics_targeted.py is in the same directory")
        return
    
    # Create exporter instance
    exporter = ContainerMetricsExporter()
    
    print("🔧 Configuration:")
    print(f"   Prometheus URL: {exporter.prom_url}")
    print("   Target containers:")
    for service, names in exporter.target_containers.items():
        print(f"     {service}: {', '.join(names)}")
    
    print("\n🚀 Starting export...")
    
    # Export last 15 minutes of data
    output_file = exporter.export_metrics(seconds=900)  # 15 minutes
    
    if output_file:
        print(f"\n✅ Success! Data exported to: {output_file}")
        
        # Quick analysis
        try:
            import pandas as pd
            df = pd.read_csv(output_file)
            
            print(f"\n📊 Quick Stats:")
            print(f"   Total records: {len(df):,}")
            print(f"   Containers: {df['container_name'].nunique()}")
            print(f"   Metrics: {df['metric_name'].nunique()}")
            print(f"   Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            print(f"\n📦 Container breakdown:")
            for container, count in df['container_name'].value_counts().items():
                percentage = (count / len(df)) * 100
                print(f"   {container}: {count:,} records ({percentage:.1f}%)")
                
        except ImportError:
            print("   (Install pandas for detailed analysis)")
        except Exception as e:
            print(f"   Analysis error: {e}")
    else:
        print("❌ Export failed")


if __name__ == '__main__':
    main()