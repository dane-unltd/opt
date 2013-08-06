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
	return &Armijo{IterMax: 1000}
}

func (s *Armijo) Solve(m *Model) error {
	var err error = nil

	tStart := time.Now()

	beta := 0.5

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
		err = errors.New("Armijo: No progress possible")
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
			m.Time = time.Since(tStart)
			m.DoCallbacks()

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
			m.Time = time.Since(tStart)
			m.DoCallbacks()

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
		err = errors.New("Armijo: Maximum number of Iterations reached")
	}

	if m.ObjX >= m.ObjLB {
		println(m.DerivLB, step, maxStep)
		err = errors.New("Armijo: No progress made")
	}

	return err
}
