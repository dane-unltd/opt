package linesearch

import "math"
import "errors"

//Exact line search for strictly quasi-convex functions
type Quadratic struct {
	Tol     float64
	IterMax int
}

func (s Quadratic) Solve(m *Model) (*Result, error) {
	var err error = nil

	if m.UB <= m.LB || math.IsNaN(m.UB) {
		m.UB = math.Inf(1)
	}

	x1 := m.LB
	f1 := m.LBF
	if math.IsNaN(f1) {
		f1 = m.F(x1)
	}

	f2, f3, fNew := 0.0, 0.0, 0.0
	x2, x3, xNew := 0.0, 0.0, 0.0

	iter := 0

	if math.IsInf(m.UB, 1) {
		xNew = m.X
		if math.IsNaN(xNew) {
			xNew = 1
		}
		fNew = m.F(xNew)

		if fNew < f1 {
			f2 = fNew
			x2 = xNew

			x3 = 2 * x2
			f3 = m.F(x3)
			for ; iter < s.IterMax && f3 <= f2; iter++ {
				x3 *= 2
				f3 = m.F(x3)

			}
		} else {
			f3 = fNew
			x3 = xNew

			x2 = 0.5 * x3
			f2 = m.F(x2)
			for ; iter < s.IterMax && f2 >= f1; iter++ {
				x2 *= 0.5
				f2 = m.F(x2)
			}
		}
	} else {
		x3 = m.UB
		f3 = m.F(x3)
		if f3 < f1 {
			x2 = x3 - s.Tol
			f2 = m.F(x2)
			if f2 >= f3 {
				xNew = x3
				fNew = f3
				goto done
			}
		} else {
			x2 = 0.5 * x3
			f2 = m.F(x2)
			for ; iter < s.IterMax && f2 >= f1; iter++ {
				x2 *= 0.5
				f2 = m.F(x2)
			}
		}
	}

	for ; x3-x1 > s.Tol && iter < s.IterMax; iter++ {
		xNew = -0.5 * (x3*x3*(f1-f2) + x2*x2*(f3-f1) + x1*x1*(f2-f3)) /
			(x3*(f2-f1) + x2*(f1-f3) + x1*(f3-f2))
		fNew = m.F(xNew)
		if xNew > x2 {
			if fNew >= f2 {
				x3 = xNew
			} else {
				x1 = x2
				x2 = xNew
				f2 = fNew
			}
		} else {
			if fNew >= f2 {
				x1 = xNew
			} else {
				x3 = x2
				x2 = xNew
				f2 = fNew
			}
		}
	}

done:
	if iter == s.IterMax {
		err = errors.New("Maximum number of Iterations reached")
	}

	m.UB = x3
	m.LB = x1
	m.LBF = f1
	m.X = x2

	res := &Result{X: xNew, F: fNew, G: math.NaN(), Iter: iter}
	return res, err
}
