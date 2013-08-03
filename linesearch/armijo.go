package linesearch

import "math"
import "errors"

//Inexact line search using Armijo's rule.
type Armijo struct {
	IterMax int
}

func (s Armijo) Solve(m *Model) (*Result, error) {
	var err error = nil

	beta := 0.5

	if m.UB <= m.LB || math.IsNaN(m.UB) {
		m.UB = math.Inf(1)
	}

	f0 := m.LBF
	g0 := m.LBG
	x0 := m.LB
	step := m.X - x0
	maxStep := m.UB - x0

	if math.IsNaN(g0) {
		g0 = m.G(x0)
	}
	if math.IsNaN(f0) {
		f0 = m.F(x0)
	}
	if math.IsNaN(step) || step < 0 || step > maxStep {
		if math.IsInf(m.UB, 1) {
			step = 1
		} else {
			step = maxStep
		}
	}

	if g0 > 0 {
		err = errors.New("No progress possible")
		return nil, err
	}

	fNew := m.F(x0 + step)

	iter := 0
	if fNew-f0 > 0.5*g0*step {
		fPrev := fNew
		step *= beta
		for {
			iter++
			if iter == s.IterMax {
				break
			}
			fNew = m.F(x0 + step)
			if fNew-f0 <= 0.5*g0*step {
				if fPrev < fNew {
					step /= beta
					fNew = fPrev
				}
				break
			}
			fPrev = fNew
			step *= beta
		}
	} else {
		fPrev := fNew
		if step == maxStep {
			goto done
		}
		step /= beta
		if step > maxStep {
			step = maxStep
		}
		for {
			iter++
			if iter == s.IterMax {
				break
			}
			fNew = m.F(x0 + step)
			if fNew-f0 > 0.5*g0*step {
				if fPrev < fNew {
					step *= beta
					fNew = fPrev
				}
				break
			}
			fPrev = fNew
			if step == maxStep {
				goto done
			}
			step /= beta
			if step > maxStep {
				step = maxStep
			}
		}
	}

done:
	if iter == s.IterMax {
		err = errors.New("Maximum number of Iterations reached")
	}

	//refine model
	m.X = x0 + step
	m.LBG = g0
	m.LBF = f0

	r := &Result{X: x0 + step, F: fNew, G: math.NaN(), Iter: iter}
	return r, err
}
