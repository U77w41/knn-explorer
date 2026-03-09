import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TheoryModule } from '@/components/TheoryModule';
import { ClassificationPlayground } from '@/components/ClassificationPlayground';
import { RegressionPlayground } from '@/components/RegressionPlayground';
import { PreprocessingModule } from '@/components/PreprocessingModule';
import {
  BookOpen,
  PlayCircle,
  TrendingUp,
  Scale,
  Github,
  Sparkles,
} from 'lucide-react';

/**
 * KNN Interactive Learning Application
 * A comprehensive, client-side educational tool for understanding
 * K-Nearest Neighbors algorithm for classification and regression
 */
const Index = () => {
  const [activeTab, setActiveTab] = useState('theory');

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Header */}
      <header className="relative overflow-hidden border-b border-border/50">
        {/* Gradient background effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/5" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        
        <div className="container mx-auto px-4 py-8 relative">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg bg-primary/20 animate-pulse-glow">
                  <Sparkles className="w-6 h-6 text-primary" />
                </div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
                  <span className="bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
                    K-Nearest Neighbors
                  </span>
                </h1>
              </div>
              <p className="text-muted-foreground max-w-xl">
                Interactive visual learning tool for understanding KNN classification 
                and regression. Drag sliders, click plots, and watch the algorithm decide.
              </p>
            </div>
            
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="px-3 py-1">
                100% Client-Side
              </Badge>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm">
                  <Github className="w-4 h-4 mr-2" />
                  GitHub
                </Button>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          {/* Tab Navigation */}
          <TabsList className="grid w-full max-w-2xl mx-auto grid-cols-4 bg-muted/50 p-1 h-auto">
            <TabsTrigger 
              value="theory" 
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <BookOpen className="w-4 h-4" />
              <span className="text-xs">Theory</span>
            </TabsTrigger>
            <TabsTrigger 
              value="classification"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <PlayCircle className="w-4 h-4" />
              <span className="text-xs">Classify</span>
            </TabsTrigger>
            <TabsTrigger 
              value="regression"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <TrendingUp className="w-4 h-4" />
              <span className="text-xs">Regress</span>
            </TabsTrigger>
            <TabsTrigger 
              value="preprocessing"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Scale className="w-4 h-4" />
              <span className="text-xs">Scaling</span>
            </TabsTrigger>
          </TabsList>

          {/* Theory Tab */}
          <TabsContent value="theory" className="animate-fade-in-up">
            <TheoryModule />
          </TabsContent>

          {/* Classification Tab */}
          <TabsContent value="classification" className="animate-fade-in-up">
            <ClassificationPlayground />
          </TabsContent>

          {/* Regression Tab */}
          <TabsContent value="regression" className="animate-fade-in-up">
            <RegressionPlayground />
          </TabsContent>

          {/* Preprocessing Tab */}
          <TabsContent value="preprocessing" className="animate-fade-in-up">
            <PreprocessingModule />
          </TabsContent>
        </Tabs>

        {/* Quick Tips */}
        <section className="mt-12 p-6 glass-card">
          <h2 className="section-header mb-4">Quick Guide</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <h3 className="font-semibold text-primary">🎯 Classification</h3>
              <p className="text-sm text-muted-foreground">
                Drag the K slider to see how decision boundaries change. 
                K=1 creates complex, overfit boundaries. Large K creates smooth, underfit ones.
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-secondary">📈 Regression</h3>
              <p className="text-sm text-muted-foreground">
                Watch the orange prediction line respond to K. 
                K=1 creates a step function; larger K smooths the curve.
              </p>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-accent">⚖️ Feature Scaling</h3>
              <p className="text-sm text-muted-foreground">
                See how unscaled features (Age vs Salary) produce terrible results. 
                Apply Min-Max or Z-Score scaling to fix it.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>
            Built with React, Plotly.js, and KaTeX. 
            <span className="text-primary mx-2">•</span>
            Zero backend required.
            <span className="text-primary mx-2">•</span>
            Deploy anywhere.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
