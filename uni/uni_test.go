package uni

import (
	"math"
	"testing"
)

// http://www.wolframalpha.com/input/?i=0.3+*+exp%28+-+3+%28x-1%29%29+%2B+exp%28x-1%29
type SumExpStruct struct{}

func (s SumExpStruct) F(x float64) (f float64) {
	c1 := 0.3
	c2 := 3.0
	f = c1*math.Exp(-c2*(x-1)) + math.Exp((x - 1))
	return f
}

func (s SumExpStruct) DF(x float64) float64 {
	c1 := 0.3
	c2 := 3.0
	return -c1*c2*math.Exp(-c2*(x-1)) + math.Exp((x - 1))
}

func (s SumExpStruct) FdF(x float64) (f, d float64) {
	c1 := 0.3
	c2 := 3.0
	f = c1*math.Exp(-c2*(x-1)) + math.Exp((x - 1))
	d = -c1*c2*math.Exp(-c2*(x-1)) + math.Exp((x - 1))
	return f, d
}

func (s SumExpStruct) Name() string {
	return "SumExp"
}

func (s SumExpStruct) OptVal() float64 {
	return 1.298671661900395685896941595
}

func (s SumExpStruct) OptLoc() float64 {
	return 0.9736598710855434246931247548
}

func TestUni(t *testing.T) {
	fun := SumExpStruct{}

	in := NewSolution()
	in.Set(0.5)
	in.ObjLower, in.DerivLower = fun.FdF(in.XLower)

	r := NewQuadratic().OptimizeF(fun, in, Accuracy(1e-3))

	t.Log(r.XLower, r.X, r.XUpper, r.Status)
	t.Log(r.Obj - fun.OptVal())
	t.Log(r.FunEvals, r.Status)

	if math.Abs(r.X-fun.OptLoc()) > 0.01 {
		t.Fail()
	}

	r = NewBacktracking().OptimizeF(fun, in)
	t.Log(r.FunEvals, r.Status)

	if r.Obj >= r.ObjLower {
		t.Fail()
	}

	r = NewCubic().OptimizeFdF(fun, in, Accuracy(1e-3))

	t.Log(r.XLower, r.X, r.XUpper, r.Status)
	t.Log(r.Obj - fun.OptVal())
	t.Log(r.FunEvals, r.Status)

	if math.Abs(r.X-fun.OptLoc()) > 0.01 {
		t.Fail()
	}
}
