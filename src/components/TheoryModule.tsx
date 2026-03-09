import { useState } from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BookOpen, Calculator, AlertTriangle } from 'lucide-react';

/**
 * Theory Module - Educational content about KNN algorithm
 * Includes interactive distance metric explanations
 */
export function TheoryModule() {
  const [minkowskiP, setMinkowskiP] = useState(2);
  const [point1] = useState([1, 2]);
  const [point2] = useState([4, 6]);

  // Calculate distances for demonstration
  const euclidean = Math.sqrt(
    Math.pow(point2[0] - point1[0], 2) + Math.pow(point2[1] - point1[1], 2)
  ).toFixed(3);
  
  const manhattan = (
    Math.abs(point2[0] - point1[0]) + Math.abs(point2[1] - point1[1])
  ).toFixed(3);
  
  const minkowski = Math.pow(
    Math.pow(Math.abs(point2[0] - point1[0]), minkowskiP) +
    Math.pow(Math.abs(point2[1] - point1[1]), minkowskiP),
    1 / minkowskiP
  ).toFixed(3);

  return (
    <div className="space-y-8 animate-fade-in-up">
      {/* Algorithm Overview */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 section-header">
            <BookOpen className="w-6 h-6 text-primary" />
            What is K-Nearest Neighbors?
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground leading-relaxed">
            K-Nearest Neighbors (KNN) is a <span className="text-primary font-semibold">non-parametric</span>, 
            <span className="text-secondary font-semibold"> lazy learning</span> algorithm used for both 
            classification and regression tasks.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4 mt-4">
            <div className="p-4 rounded-lg bg-muted/30 border border-border/50">
              <h4 className="font-semibold text-primary mb-2">Non-parametric</h4>
              <p className="text-sm text-muted-foreground">
                Makes no assumptions about the underlying data distribution. 
                The model structure is determined from the data itself.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-muted/30 border border-border/50">
              <h4 className="font-semibold text-secondary mb-2">Lazy Learning</h4>
              <p className="text-sm text-muted-foreground">
                No explicit training phase. All computation is deferred until prediction time, 
                making it memory-intensive but flexible.
              </p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-primary/5 rounded-lg border border-primary/20">
            <h4 className="font-semibold mb-2">How it works:</h4>
            <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
              <li>Store all training data points</li>
              <li>For a new query point, calculate distance to all training points</li>
              <li>Find the <InlineMath math="K" /> nearest neighbors</li>
              <li>
                <strong>Classification:</strong> Vote among neighbors to determine class
              </li>
              <li>
                <strong>Regression:</strong> Average the values of neighbors
              </li>
            </ol>
          </div>
        </CardContent>
      </Card>

      {/* Distance Metrics */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 section-header">
            <Calculator className="w-6 h-6 text-primary" />
            Distance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground mb-6">
            The choice of distance metric significantly affects KNN's performance. 
            Here are the most common metrics:
          </p>

          <Tabs defaultValue="euclidean" className="w-full">
            <TabsList className="grid w-full grid-cols-3 bg-muted/50">
              <TabsTrigger value="euclidean">Euclidean</TabsTrigger>
              <TabsTrigger value="manhattan">Manhattan</TabsTrigger>
              <TabsTrigger value="minkowski">Minkowski</TabsTrigger>
            </TabsList>

            <TabsContent value="euclidean" className="mt-4">
              <div className="space-y-4">
                <div className="p-4 bg-muted/20 rounded-lg">
                  <BlockMath math="d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}" />
                </div>
                <p className="text-sm text-muted-foreground">
                  The straight-line distance between two points. Also known as L2 norm.
                  Best for continuous, normally distributed features.
                </p>
                <div className="p-3 bg-accent/10 rounded border border-accent/30">
                  <p className="text-sm">
                    <span className="text-accent font-semibold">Example:</span> Points 
                    <InlineMath math={`(${point1[0]}, ${point1[1]})`} /> and 
                    <InlineMath math={`(${point2[0]}, ${point2[1]})`} />
                  </p>
                  <p className="text-lg font-mono mt-2 text-primary">
                    Distance = {euclidean}
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="manhattan" className="mt-4">
              <div className="space-y-4">
                <div className="p-4 bg-muted/20 rounded-lg">
                  <BlockMath math="d(p, q) = \sum_{i=1}^{n} |p_i - q_i|" />
                </div>
                <p className="text-sm text-muted-foreground">
                  Sum of absolute differences. Also known as L1 norm or "taxicab distance".
                  More robust to outliers than Euclidean.
                </p>
                <div className="p-3 bg-secondary/10 rounded border border-secondary/30">
                  <p className="text-sm">
                    <span className="text-secondary font-semibold">Example:</span> Points 
                    <InlineMath math={`(${point1[0]}, ${point1[1]})`} /> and 
                    <InlineMath math={`(${point2[0]}, ${point2[1]})`} />
                  </p>
                  <p className="text-lg font-mono mt-2 text-secondary">
                    Distance = {manhattan}
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="minkowski" className="mt-4">
              <div className="space-y-4">
                <div className="p-4 bg-muted/20 rounded-lg">
                  <BlockMath math="d(p, q) = \left( \sum_{i=1}^{n} |p_i - q_i|^p \right)^{1/p}" />
                </div>
                <p className="text-sm text-muted-foreground">
                  Generalization of Euclidean (p=2) and Manhattan (p=1) distances.
                  The parameter p controls the "shape" of the distance metric.
                </p>
                
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground w-20">
                      p = {minkowskiP.toFixed(1)}
                    </span>
                    <Slider
                      value={[minkowskiP]}
                      onValueChange={(v) => setMinkowskiP(v[0])}
                      min={0.5}
                      max={5}
                      step={0.1}
                      className="flex-1"
                    />
                  </div>
                  <div className="p-3 bg-primary/10 rounded border border-primary/30">
                    <p className="text-sm">
                      <span className="text-primary font-semibold">Interactive:</span> Points 
                      <InlineMath math={`(${point1[0]}, ${point1[1]})`} /> and 
                      <InlineMath math={`(${point2[0]}, ${point2[1]})`} />
                    </p>
                    <p className="text-lg font-mono mt-2 text-primary">
                      Distance = {minkowski}
                    </p>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Curse of Dimensionality */}
      <Card className="glass-card border-destructive/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-destructive">
            <AlertTriangle className="w-6 h-6" />
            The Curse of Dimensionality
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            As the number of features (dimensions) increases, distance metrics become 
            increasingly meaningless. This is a critical limitation of KNN.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">Why it happens:</h4>
              <p className="text-sm text-muted-foreground">
                In high dimensions, all points become approximately equidistant from each other. 
                The ratio of the nearest to farthest neighbor approaches 1:
              </p>
              <div className="mt-3">
                <BlockMath math="\lim_{d \to \infty} \frac{d_{max} - d_{min}}{d_{min}} \to 0" />
              </div>
            </div>
            <div className="p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">Solutions:</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="text-primary">Feature selection</span> - Keep only relevant features</li>
                <li>• <span className="text-primary">Dimensionality reduction</span> - PCA, t-SNE</li>
                <li>• <span className="text-primary">Feature scaling</span> - Normalize all features</li>
                <li>• <span className="text-primary">More data</span> - Exponentially more samples needed</li>
              </ul>
            </div>
          </div>

          <div className="p-4 bg-destructive/10 rounded-lg border border-destructive/30 mt-4">
            <p className="text-sm">
              <strong className="text-destructive">Rule of thumb:</strong> KNN works best with 
              fewer than 20 features after proper preprocessing. For higher dimensions, 
              consider other algorithms or apply dimensionality reduction first.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
