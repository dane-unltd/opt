package multi

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas"
	"github.com/gonum/blas/cblas"
	"github.com/gonum/blas/dbw"
)

var cops struct {
	cblas.Blas
}

func init() {
	dbw.Register(cops)
}

func TestExampleModel(t *testing.T) {
	//badly conditioned Hessian leads to zig-zagging of the steepest descent
	//algorithm
	condNo := 100.0
	optSol := dbw.NewVector([]float64{1, 2})

	A := dbw.NewGeneral(2, 2, []float64{condNo, 0, 0, 1})
	b := dbw.NewVector([]float64{-2 * optSol.Data[0] * condNo,
		-2 * optSol.Data[1]})
	c := -0.5 * dbw.Dot(b, optSol)

	//define objective function
	fun := opt.NewQuadratic(A, b.Data, c)

	//set inital solution estimate
	sol := NewSolution(make([]float64, 2))

	//set termination parameters
	p := NewParams()
	p.IterMax = 5

	//Use steepest descent solver to solve the model
	solver := NewSearchBased(SteepestDescent{}, uni.NewBacktracking())
	result := solver.OptimizeFdF(fun, sol, p, GradConv{1e-3}, NewDisplay(1))

	fmt.Println("x =", result.X)
	//should be [1,2], but because of the bad conditioning we made little
	//progress in the second dimension

	//Use a BFGS solver to refine the result:
	solver = NewSearchBased(new(LBFGS), uni.NewCubic())
	result = solver.OptimizeFdF(fun, &result.Solution, p, GradConv{1e-3}, NewDisplay(1))

	fmt.Println("x =", result.X)
}

func TestQuadratic(t *testing.T) {
	dbw.Register(cops)

	n := 5

	xStar := dbw.NewVector(make([]float64, n))
	for i := range xStar.Data {
		xStar.Data[i] = 1
	}
	A := dbw.NewGeneral(n, n, make([]float64, n*n))
	for i := range A.Data {
		A.Data[i] = rand.NormFloat64()
	}
	AtA := dbw.NewGeneral(n, n, make([]float64, n*n))
	dbw.Gemm(blas.Trans, blas.NoTrans, 1, A, A, 0, AtA)

	bTmp := dbw.NewVector(make([]float64, n))
	dbw.Gemv(blas.NoTrans, 1, A, xStar, 0, bTmp)
	b := dbw.NewVector(make([]float64, n))
	dbw.Gemv(blas.Trans, -2, A, bTmp, 0, b)

	c := dbw.Dot(bTmp, bTmp)

	//Define input arguments
	obj := opt.NewQuadratic(AtA, b.Data, c)
	sol := NewSolution(make([]float64, n))

	//Steepest descent with Backtracking
	stDesc := NewSearchBased(SteepestDescent{}, uni.NewBacktracking())
	res1 := stDesc.OptimizeFdF(obj, sol, GradConv{1e-6}, NewDisplay(100))

	t.Log(res1.Obj, res1.FunEvals, res1.GradEvals, res1.Status)

	//Steepest descent with Cubic
	stDesc = NewSearchBased(SteepestDescent{}, uni.NewCubic())
	res2 := stDesc.OptimizeFdF(obj, sol, GradConv{1e-6}, NewDisplay(100))

	t.Log(res2.Obj, res2.FunEvals, res2.GradEvals, res2.Status)

	//LBFGS with Cubic
	lbfgs := NewSearchBased(new(LBFGS), uni.NewCubic())
	res3 := lbfgs.OptimizeFdF(obj, sol, GradConv{1e-6}, NewDisplay(1))

	t.Log(res3.Obj, res3.FunEvals, res3.GradEvals, res3.Status)

	//constrained problems (constraints described as projection)
	projGrad := NewProjGrad()

	res4 := projGrad.OptimizeFProj(obj, opt.RealPlus{}, sol, NewDeltaXConv(1e-6), NewDisplay(100))

	t.Log(res4.Obj, res4.FunEvals, res4.GradEvals, res4.Status)

	if math.Abs(res1.Obj) > 0.01 {
		t.Fail()
	}
	if math.Abs(res2.Obj) > 0.01 {
		t.Fail()
	}
	if math.Abs(res3.Obj) > 0.01 {
		t.Fail()
	}
	if math.Abs(res4.Obj) > 0.01 {
		t.Fail()
	}
}

func TestRosenbrock(t *testing.T) {
	dbw.Register(cops)

	n := 10
	scale := 10.0

	xInit := make([]float64, n)
	for i := range xInit {
		xInit[i] = scale * rand.Float64()
	}

	//Define input arguments
	obj := opt.Rosenbrock{}
	sol := NewSolution(xInit)

	//Steepest descent with Backtracking
	stDesc := NewSearchBased(SteepestDescent{}, uni.NewBacktracking())
	res1 := stDesc.OptimizeFdF(obj, sol, GradConv{1e-6},
		Termination{10000, time.Minute}, NewDisplay(1000))

	t.Log(res1.Obj, res1.FunEvals, res1.GradEvals, res1.Status)

	//Steepest descent with Qubic
	stDesc = NewSearchBased(SteepestDescent{}, uni.NewCubic())
	res2 := stDesc.OptimizeFdF(obj, sol, GradConv{1e-6}, NewDisplay(1000))

	t.Log(res2.Obj, res2.FunEvals, res2.GradEvals, res2.Status)

	//LBFGS with Cubic
	lbfgs := NewSearchBased(new(LBFGS), uni.NewCubic())
	res3 := lbfgs.OptimizeFdF(obj, sol, GradConv{1e-6}, NewDisplay(10))

	t.Log(res3.Obj, res3.FunEvals, res3.GradEvals, res3.Status)

	//LBFGS with Backtracking
	lbfgs = NewSearchBased(new(LBFGS), uni.NewBacktracking())
	res5 := lbfgs.OptimizeFdF(obj, sol, GradConv{1e-6}, NewDisplay(10))

	t.Log(res5.Obj, res5.FunEvals, res5.GradEvals, res5.Status)

	if math.Abs(res1.Obj) > 0.01 {
		t.Fail()
	}
	if math.Abs(res2.Obj) > 0.01 {
		t.Fail()
	}
	if math.Abs(res3.Obj) > 0.01 {
		t.Fail()
	}
	if math.Abs(res5.Obj) > 0.01 {
		t.Fail()
	}
}

type rb struct {
	opt.Rosenbrock
}

func (r rb) FdF() {}

type rosTest struct{}

func (r rosTest) F(x []float64) float64 {
	return math.Pow(x[0]-2, 4) + math.Pow(x[0]-2*x[1], 2)
}

func TestSolve(t *testing.T) {
	dbw.Register(cops)

	n := 10
	scale := 10.0

	xInit := make([]float64, n)
	for i := range xInit {
		xInit[i] = scale * rand.Float64()
	}
	sol := NewSolution(xInit)

	result := OptimizeF(opt.Rosenbrock{}, sol, nil, NewDeltaXConv(1e-6), NewDisplay(10))

	t.Log(result.Status, result.Obj, result.Iter)
	if math.Abs(result.Obj) > 0.1 {
		t.Fail()
	}

	params := NewParams()
	params.IterMax = 100000

	result = OptimizeFProj(opt.Rosenbrock{}, opt.RealPlus{}, sol,
		params, NewDisplay(10000))
	t.Log(result.Status, result.Obj, result.Iter)

	params.Accuracy = 1e-5
	params.IterMax = 1000

	result = OptimizeF(rb{}, sol, params, NewDisplay(3))
	t.Log(result.Status, result.Obj, result.Iter, result.X)

	xInit = []float64{0, 3}
	sol.SetX(xInit, false)
	result = OptimizeF(rosTest{}, sol, params, NewDisplay(10))
	t.Log(result.Status, result.Obj, result.Iter)
}
