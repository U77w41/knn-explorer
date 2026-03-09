import { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { InlineMath, BlockMath } from 'react-katex';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { Upload, Scale, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { DataFrame } from 'danfojs';
import {
  DataPoint,
  minMaxScale,
  standardScale,
  generateDecisionBoundaryGrid,
  getDistanceFunction,
  calculateAccuracy,
} from '@/lib/knnEngine';

type ScalingMethod = 'none' | 'minmax' | 'standard';

/**
 * Data Preprocessing Module - Demonstrates feature scaling importance
 * Shows dramatic accuracy improvement when features are scaled
 */
export function PreprocessingModule() {
  const [rawData, setRawData] = useState<DataPoint[]>([]);
  const [scaledData, setScaledData] = useState<DataPoint[]>([]);
  const [scalingMethod, setScalingMethod] = useState<ScalingMethod>('none');
  const [isLoaded, setIsLoaded] = useState(false);
  const [fileName, setFileName] = useState<string>('');

  // Generate sample unscaled data for demonstration
  const generateSampleData = useCallback(() => {
    const data: DataPoint[] = [];
    
    // Class 0: Younger with lower salary
    for (let i = 0; i < 50; i++) {
      const age = 22 + Math.random() * 15; // 22-37
      const salary = 35000 + Math.random() * 25000; // 35k-60k
      data.push({ features: [age, salary], label: 0 });
    }
    
    // Class 1: Older with higher salary
    for (let i = 0; i < 50; i++) {
      const age = 35 + Math.random() * 25; // 35-60
      const salary = 55000 + Math.random() * 45000; // 55k-100k
      data.push({ features: [age, salary], label: 1 });
    }
    
    setRawData(data);
    setScaledData(data);
    setScalingMethod('none');
    setIsLoaded(true);
    setFileName('sample_data.csv');
  }, []);

  // Handle CSV file upload
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const lines = text.trim().split('\n');
      const headers = lines[0].split(',');
      
      const data: DataPoint[] = [];
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(Number);
        if (values.length >= 3 && values.every(v => !isNaN(v))) {
          data.push({
            features: [values[0], values[1]],
            label: values[2],
          });
        }
      }
      
      if (data.length > 0) {
        setRawData(data);
        setScaledData(data);
        setScalingMethod('none');
        setIsLoaded(true);
        setFileName(file.name);
      }
    } catch (error) {
      console.error('Error parsing CSV:', error);
    }
  }, []);

  // Apply scaling
  const applyScaling = useCallback((method: ScalingMethod) => {
    if (rawData.length === 0) return;
    
    setScalingMethod(method);
    
    if (method === 'none') {
      setScaledData(rawData);
      return;
    }
    
    const features = rawData.map(p => p.features);
    let scaled: number[][];
    
    if (method === 'minmax') {
      scaled = minMaxScale(features).scaled;
    } else {
      scaled = standardScale(features).scaled;
    }
    
    const newData = rawData.map((p, i) => ({
      features: scaled[i],
      label: p.label,
    }));
    
    setScaledData(newData);
  }, [rawData]);

  // Calculate decision boundary for current data
  const distanceFn = useMemo(() => getDistanceFunction('euclidean'), []);
  
  const { gridData, bounds, accuracy } = useMemo(() => {
    const activeData = scalingMethod === 'none' ? rawData : scaledData;
    if (activeData.length < 4) {
      return { gridData: null, bounds: null, accuracy: 0 };
    }
    
    const xs = activeData.map(p => p.features[0]);
    const ys = activeData.map(p => p.features[1]);
    const padding = scalingMethod === 'none' ? 
      [(Math.max(...xs) - Math.min(...xs)) * 0.1, (Math.max(...ys) - Math.min(...ys)) * 0.1] :
      [0.2, 0.2];
    
    const bounds = {
      xMin: Math.min(...xs) - padding[0],
      xMax: Math.max(...xs) + padding[0],
      yMin: Math.min(...ys) - padding[1],
      yMax: Math.max(...ys) + padding[1],
    };
    
    const gridData = generateDecisionBoundaryGrid(
      activeData, 5, distanceFn, 'uniform',
      [bounds.xMin, bounds.xMax],
      [bounds.yMin, bounds.yMax],
      30
    );
    
    // Calculate leave-one-out cross-validation accuracy approximation
    const accuracy = calculateAccuracy(activeData, activeData, 5, distanceFn, 'uniform');
    
    return { gridData, bounds, accuracy };
  }, [rawData, scaledData, scalingMethod, distanceFn]);

  // Separate by class
  const activeData = scalingMethod === 'none' ? rawData : scaledData;
  const class0 = activeData.filter(p => p.label === 0);
  const class1 = activeData.filter(p => p.label === 1);

  // Plot data
  const traces: object[] = [];
  
  if (gridData && bounds) {
    traces.push({
      type: 'contour',
      x: gridData.x,
      y: gridData.y,
      z: gridData.z,
      colorscale: [
        [0, 'rgba(59, 130, 246, 0.25)'],
        [1, 'rgba(249, 115, 22, 0.25)'],
      ],
      showscale: false,
      contours: { coloring: 'heatmap' },
      hoverinfo: 'skip',
    });
  }
  
  traces.push({
    type: 'scatter',
    mode: 'markers',
    x: class0.map(p => p.features[0]),
    y: class0.map(p => p.features[1]),
    marker: { size: 8, color: 'hsl(217, 91%, 60%)', line: { width: 1, color: 'hsl(217, 91%, 40%)' } },
    name: 'Class 0',
  });
  
  traces.push({
    type: 'scatter',
    mode: 'markers',
    x: class1.map(p => p.features[0]),
    y: class1.map(p => p.features[1]),
    marker: { size: 8, color: 'hsl(32, 95%, 55%)', line: { width: 1, color: 'hsl(32, 95%, 40%)' } },
    name: 'Class 1',
  });

  const layout = {
    autosize: true,
    height: 400,
    margin: { t: 30, r: 20, b: 50, l: 60 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
    font: { color: '#94A3B8' },
    xaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      title: scalingMethod === 'none' ? 'Age' : 'Age (scaled)',
      range: bounds ? [bounds.xMin, bounds.xMax] : undefined,
    },
    yaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      title: scalingMethod === 'none' ? 'Salary ($)' : 'Salary (scaled)',
      range: bounds ? [bounds.yMin, bounds.yMax] : undefined,
    },
    legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(15, 23, 42, 0.8)' },
    title: {
      text: scalingMethod === 'none' 
        ? 'Raw Data (Salary dominates distance!)' 
        : `Scaled Data (${scalingMethod === 'minmax' ? 'Min-Max' : 'Z-Score'})`,
      font: { size: 14, color: scalingMethod === 'none' ? '#EF4444' : '#22C55E' },
    },
  };

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="section-header flex items-center gap-3">
          <Scale className="w-6 h-6 text-accent" />
          Feature Scaling Demonstration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Explanation */}
        <Alert className="border-accent/30 bg-accent/5">
          <AlertTriangle className="h-4 w-4 text-accent" />
          <AlertTitle className="text-accent">Why Scaling Matters</AlertTitle>
          <AlertDescription className="text-muted-foreground">
            KNN uses distance metrics. Features with larger ranges (like salary: $35k-$100k) 
            will dominate over features with smaller ranges (age: 22-60). 
            <strong className="text-foreground"> This makes age almost irrelevant!</strong>
          </AlertDescription>
        </Alert>

        {/* Data Input */}
        <div className="flex flex-wrap gap-4 items-center">
          <Button onClick={generateSampleData} variant="outline">
            Generate Sample Data
          </Button>
          <div className="relative">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
            <Button variant="outline">
              <Upload className="w-4 h-4 mr-2" />
              Upload CSV
            </Button>
          </div>
          {fileName && (
            <Badge variant="secondary">{fileName} ({rawData.length} rows)</Badge>
          )}
        </div>

        {isLoaded && (
          <>
            {/* Scaling Options */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Button
                variant={scalingMethod === 'none' ? 'destructive' : 'outline'}
                onClick={() => applyScaling('none')}
                className="h-auto py-4"
              >
                <div className="text-left">
                  <div className="font-semibold">No Scaling</div>
                  <div className="text-xs opacity-80">Raw features</div>
                </div>
              </Button>
              
              <Button
                variant={scalingMethod === 'minmax' ? 'default' : 'outline'}
                onClick={() => applyScaling('minmax')}
                className="h-auto py-4"
              >
                <div className="text-left">
                  <div className="font-semibold">Min-Max Scaling</div>
                  <div className="text-xs opacity-80">Scale to [0, 1]</div>
                </div>
              </Button>
              
              <Button
                variant={scalingMethod === 'standard' ? 'default' : 'outline'}
                onClick={() => applyScaling('standard')}
                className="h-auto py-4"
              >
                <div className="text-left">
                  <div className="font-semibold">Z-Score (Standard)</div>
                  <div className="text-xs opacity-80">μ=0, σ=1</div>
                </div>
              </Button>
            </div>

            {/* Formula Display */}
            {scalingMethod !== 'none' && (
              <div className="p-4 bg-muted/20 rounded-lg">
                {scalingMethod === 'minmax' ? (
                  <div>
                    <h4 className="text-sm font-semibold mb-2">Min-Max Normalization:</h4>
                    <BlockMath math="x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}" />
                  </div>
                ) : (
                  <div>
                    <h4 className="text-sm font-semibold mb-2">Z-Score Standardization:</h4>
                    <BlockMath math="z = \frac{x - \mu}{\sigma}" />
                  </div>
                )}
              </div>
            )}

            {/* Visualization */}
            <div className="plotly-container">
              <Plot
                data={traces as any}
                layout={layout as any}
                config={{ responsive: true, displayModeBar: true }}
                style={{ width: '100%' }}
              />
            </div>

            {/* Results Comparison */}
            <div className="grid grid-cols-2 gap-4">
              <div className={`p-4 rounded-lg border ${scalingMethod === 'none' ? 'border-destructive/50 bg-destructive/10' : 'border-border/30 bg-muted/20'}`}>
                <div className="flex items-center gap-2 mb-2">
                  {scalingMethod === 'none' ? (
                    <AlertTriangle className="w-5 h-5 text-destructive" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 text-muted-foreground" />
                  )}
                  <h4 className="font-semibold">Raw Data Analysis</h4>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Age range: 22 - 60</li>
                  <li>• Salary range: $35k - $100k</li>
                  <li>• <span className="text-destructive font-semibold">Salary dominates distances!</span></li>
                </ul>
              </div>
              
              <div className={`p-4 rounded-lg border ${scalingMethod !== 'none' ? 'border-class-2/50 bg-class-2/10' : 'border-border/30 bg-muted/20'}`}>
                <div className="flex items-center gap-2 mb-2">
                  {scalingMethod !== 'none' ? (
                    <CheckCircle2 className="w-5 h-5 text-class-2" />
                  ) : (
                    <Scale className="w-5 h-5 text-muted-foreground" />
                  )}
                  <h4 className="font-semibold">Scaled Data</h4>
                </div>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Both features: 0 - 1 (or z-score)</li>
                  <li>• Equal contribution to distance</li>
                  <li>• <span className="text-class-2 font-semibold">Fair decision boundary!</span></li>
                </ul>
              </div>
            </div>

            {/* Accuracy Display */}
            <div className="text-center p-4 bg-muted/20 rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">Training Accuracy (K=5)</p>
              <p className={`text-3xl font-bold ${accuracy > 85 ? 'text-class-2' : accuracy > 70 ? 'text-secondary' : 'text-destructive'}`}>
                {accuracy.toFixed(1)}%
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
