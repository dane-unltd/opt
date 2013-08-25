// +build ignore

//objective function
type F interface {
   F(x float64) float64
}

//with derivative
type FdF interface {
  F
  DF(x float64) float64
  FdF(x float64) (f,d float64)]
}

//Notify gets called by a solver when a new solution estimate is calculated
//(So usually every iteration)
type Observer interface {
  Notify(*Result)
}

type FOptimizer interface {
  Optimize(F,*Solution,*Params, ...Observer)*Result
}

type FdFOptimizer interface {
  Optimize(FdF, *Solution, *Params, ...Observer)*Result
}

//Global functions which choose an solver automatically (or try all solvers in parallel)
OptimizeF(F,*Solution,*Params, ...Observer)*Result
OptimizeFdF(FdF, *Solution, *Params, ...Observer)*Result

//Stores an estimate of the solution
type Solution struct {
   XLower, XUpper, X float64
   //...
}

//Stores additional information on the optimization run
type Stats struct {
   Iter int
   Time time.Duration
}

type Result {
   Solution
   Stats
}
