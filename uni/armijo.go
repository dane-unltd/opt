package uni

import (
	"errors"
	"math"
	"time"
)

//Inexact line search using Armijo's rule.
type Armijo struct {
	IterMax int
}

func NewArmijo() *Armijo {
	return &Armijo{IterMax: 100}
}

func (s *Armijo) Solve(m *Model) error {
	var err error = nil

	startT := time.Now()

	beta := 0.5

	if m.UB <= m.LB || math.IsNaN(m.UB) {
		m.UB = math.Inf(1)
	}

	step := m.X - m.LB
	maxStep := m.UB - m.LB

	if math.IsNaN(m.DerivLB) {
		m.DerivLB = m.Deriv(m.LB)
	}
	if math.IsNaN(m.ObjLB) {
		m.ObjLB = m.Obj(m.LB)
	}
	if math.IsNaN(step) || step <= 0 || step > maxStep {
		if math.IsInf(m.UB, 1) {
			step = 1
		} else {
			step = maxStep
		}
	}

	if m.DerivLB > 0 {
		err = errors.New("No progress possible")
		return err
	}

	m.X = m.LB + step
	m.ObjX = m.Obj(m.X)

	m.Iter = 0

	if m.ObjX-m.ObjLB > 0.5*m.DerivLB*step {
		fPrev := m.ObjX
		step *= beta
		for {
			m.Iter++
			if m.Iter == s.IterMax {
				break
			}
			m.X = m.LB + step
			m.ObjX = m.Obj(m.X)
			m.Time = time.Since(startT)
			if m.callback != nil {
				m.callback(m)
			}
			if m.ObjX-m.ObjLB <= 0.5*m.DerivLB*step {
				if fPrev < m.ObjX {
					step /= beta
					m.X = m.LB + step
					m.ObjX = fPrev
				}
				break
			}
			fPrev = m.ObjX
			step *= beta
		}
	} else {
		fPrev := m.ObjX
		if step == maxStep {
			goto done
		}
		step /= beta
		if step > maxStep {
			step = maxStep
		}
		for {
			m.Iter++
			if m.Iter == s.IterMax {
				break
			}
			m.X = m.LB + step
			m.ObjX = m.Obj(m.X)
			m.Time = time.Since(startT)
			if m.callback != nil {
				m.callback(m)
			}
			if m.ObjX-m.ObjLB > 0.5*m.DerivLB*step {
				if fPrev < m.ObjX {
					step *= beta
					m.X = m.LB + step
					m.ObjX = fPrev
				}
				break
			}
			fPrev = m.ObjX
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
	if m.Iter == s.IterMax {
		err = errors.New("Maximum number of Iterations reached")
	}

	return err
}
