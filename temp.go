package main

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Constants for the system
const (
	MaxVelocity = 10.0               // Maximum velocity for vehicles
	VehicleLen  = 4.0                // Length of each vehicle
	MaxDensity  = 0.25               // Maximum density (vehicles per unit distance)
	BrakeRatio  = 15.0               // The ratio of road length where vehicles start to brake
	Infinity    = math.MaxFloat64    // Represents infinity in Dijkstra's algorithm
)

// Node represents an intersection in the traffic system
type Node struct {
	ID          int
	Coordinates [2]float64 // Optional: X, Y coordinates for visualization
}

// Edge represents a road connecting two nodes
type Edge struct {
	ID           int
	From         *Node
	To           *Node
	Length       float64  // Length of the road
	Vehicles     []*Agent // Vehicles currently on this edge
	MaxCapacity  int      // Maximum number of vehicles allowed
}

// Calculate density of an edge (vehicles per unit length)
func (e *Edge) Density() float64 {
	return float64(len(e.Vehicles)) / e.Length
}

// Calculate weight of edge for path finding based on the paper's formula
// W(e) = ((200-α)/100) * Len(e) + 2*d(e)*l/v
func (e *Edge) Weight() float64 {
	return ((200.0 - BrakeRatio) / 100.0) * e.Length + 2.0*e.Density()*VehicleLen/MaxVelocity
}

// Agent represents a vehicle in the traffic system
type Agent struct {
	ID              int
	CurrentEdge     *Edge
	CurrentPosition float64 // Position on the current edge (0 to edge.Length)
	Velocity        float64
	Source          *Node
	Destination     *Node
	Path            []*Edge   // Planned path from source to destination
	ExpectedTime    float64   // Expected time to reach destination
	StateSequence   []State   // Sequence of states the agent goes through
	EnvyMatrix      [][]float64 // Matrix tracking envy ratios with other agents
	DeviationRate   float64   // Alpha rate for path deviation (0-1)
}

// State represents the system state
type State struct {
	ID           int
	EdgeVehicles map[int][]*Agent // Map of edge ID to vehicles on that edge
	Time         float64          // Time when this state occurred
}

// Graph represents the entire traffic network
type Graph struct {
	Nodes         []*Node
	Edges         []*Edge
	Agents        []*Agent
	CurrentState  State
	StateHistory  []State
	StateCounter  int
	Time          float64
}

// PriorityQueueItem for Dijkstra's algorithm
type PriorityQueueItem struct {
	Node     *Node
	Distance float64
	Index    int // Index in the heap
}

// PriorityQueue implementation for Dijkstra's algorithm
type PriorityQueue []*PriorityQueueItem

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].Distance < pq[j].Distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*PriorityQueueItem)
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil    // Avoid memory leak
	item.Index = -1   // For safety
	*pq = old[0 : n-1]
	return item
}

// NewGraph creates a new traffic network
func NewGraph() *Graph {
	return &Graph{
		Nodes:        make([]*Node, 0),
		Edges:        make([]*Edge, 0),
		Agents:       make([]*Agent, 0),
		StateHistory: make([]State, 0),
		CurrentState: State{
			ID:           0,
			EdgeVehicles: make(map[int][]*Agent),
			Time:         0,
		},
		StateCounter: 0,
		Time:         0,
	}
}

// AddNode adds a new node to the graph
func (g *Graph) AddNode(id int, x, y float64) *Node {
	node := &Node{
		ID:          id,
		Coordinates: [2]float64{x, y},
	}
	g.Nodes = append(g.Nodes, node)
	return node
}

// AddEdge adds a new edge to the graph
func (g *Graph) AddEdge(id int, from, to *Node, length float64, maxCapacity int) *Edge {
	edge := &Edge{
		ID:          id,
		From:        from,
		To:          to,
		Length:      length,
		Vehicles:    make([]*Agent, 0),
		MaxCapacity: maxCapacity,
	}
	g.Edges = append(g.Edges, edge)
	
	// Initialize edge vehicles in current state
	g.CurrentState.EdgeVehicles[id] = make([]*Agent, 0)
	
	return edge
}

// Find an edge that connects two nodes
func (g *Graph) FindEdge(from, to *Node) *Edge {
	for _, edge := range g.Edges {
		if edge.From == from && edge.To == to {
			return edge
		}
	}
	return nil
}

// FindNode finds a node by ID
func (g *Graph) FindNode(id int) *Node {
	for _, node := range g.Nodes {
		if node.ID == id {
			return node
		}
	}
	return nil
}

// FindEdgeByID finds an edge by ID
func (g *Graph) FindEdgeByID(id int) *Edge {
	for _, edge := range g.Edges {
		if edge.ID == id {
			return edge
		}
	}
	return nil
}

// Get outgoing edges from a node
func (g *Graph) GetOutgoingEdges(node *Node) []*Edge {
	outEdges := make([]*Edge, 0)
	for _, edge := range g.Edges {
		if edge.From == node {
			outEdges = append(outEdges, edge)
		}
	}
	return outEdges
}

// Dijkstra's algorithm for shortest path
func (g *Graph) Dijkstra(source, destination *Node) []*Edge {
	// Initialize distances
	distances := make(map[int]float64)
	previous := make(map[int]*Node)
	visited := make(map[int]bool)
	
	for _, node := range g.Nodes {
		if node == source {
			distances[node.ID] = 0
		} else {
			distances[node.ID] = Infinity
		}
		previous[node.ID] = nil
		visited[node.ID] = false
	}
	
	// Initialize priority queue
	pq := make(PriorityQueue, 0)
	heap.Init(&pq)
	
	// Add source to queue
	sourceItem := &PriorityQueueItem{
		Node:     source,
		Distance: 0,
		Index:    0,
	}
	heap.Push(&pq, sourceItem)
	
	// Main loop
	for pq.Len() > 0 {
		// Get node with smallest distance
		current := heap.Pop(&pq).(*PriorityQueueItem)
		currentNode := current.Node
		
		// If destination reached
		if currentNode == destination {
			break
		}
		
		// Skip if already visited
		if visited[currentNode.ID] {
			continue
		}
		
		visited[currentNode.ID] = true
		
		// Check all neighbors
		outEdges := g.GetOutgoingEdges(currentNode)
		for _, edge := range outEdges {
			neighborNode := edge.To
			
			// Calculate new distance
			edgeWeight := edge.Weight()
			newDist := distances[currentNode.ID] + edgeWeight
			
			// Update if better path found
			if newDist < distances[neighborNode.ID] {
				distances[neighborNode.ID] = newDist
				previous[neighborNode.ID] = currentNode
				
				// Add to queue
				heap.Push(&pq, &PriorityQueueItem{
					Node:     neighborNode,
					Distance: newDist,
				})
			}
		}
	}
	
	// Reconstruct path
	path := make([]*Edge, 0)
	current := destination
	
	// If no path found
	if previous[current.ID] == nil && current != source {
		return path // Empty path indicates no route
	}
	
	// Build path from destination to source
	for current != source {
		prev := previous[current.ID]
		edge := g.FindEdge(prev, current)
		path = append([]*Edge{edge}, path...) // Prepend to get correct order
		current = prev
	}
	
	return path
}

// AddAgent adds a new agent to the graph
func (g *Graph) AddAgent(id int, source, destination *Node, deviationRate float64) *Agent {
	agent := &Agent{
		ID:            id,
		Source:        source,
		Destination:   destination,
		Velocity:      MaxVelocity,
		CurrentPosition: 0,
		StateSequence: make([]State, 0),
		DeviationRate: deviationRate,
	}
	
	// Calculate initial path
	agent.Path = g.Dijkstra(source, destination)
	
	// Add agent to appropriate edge if path exists
	if len(agent.Path) > 0 {
		firstEdge := agent.Path[0]
		agent.CurrentEdge = firstEdge
		firstEdge.Vehicles = append(firstEdge.Vehicles, agent)
		
		// Add to current state
		g.CurrentState.EdgeVehicles[firstEdge.ID] = append(g.CurrentState.EdgeVehicles[firstEdge.ID], agent)
	}
	
	// Initialize envy matrix
	agent.InitializeEnvyMatrix(len(g.Agents) + 1)
	
	// Calculate expected time
	agent.ExpectedTime = agent.CalculateExpectedTime()
	
	g.Agents = append(g.Agents, agent)
	return agent
}

// InitializeEnvyMatrix sets up the initial envy matrix for an agent
func (a *Agent) InitializeEnvyMatrix(totalAgents int) {
	a.EnvyMatrix = make([][]float64, 1) // First row for initial expectations
	a.EnvyMatrix[0] = make([]float64, totalAgents)
}

// UpdateEnvyMatrix updates the envy matrix when agent reaches an intersection
func (a *Agent) UpdateEnvyMatrix(g *Graph) {
	// Get current number of agents
	numAgents := len(g.Agents)
	
	// Create new row for the matrix
	newRow := make([]float64, numAgents)
	
	// Calculate cost ratios
	myCost := a.CalculateExpectedTime()
	for i, otherAgent := range g.Agents {
		otherCost := otherAgent.CalculateExpectedTime()
		if otherCost > 0 {
			newRow[i] = myCost / otherCost
		} else {
			newRow[i] = 1.0 // Default to 1 if other agent has no expected time
		}
	}
	
	// Add new row to matrix
	a.EnvyMatrix = append(a.EnvyMatrix, newRow)
}

// CalculateExpectedTime estimates the time for an agent to reach destination
func (a *Agent) CalculateExpectedTime() float64 {
	if a.CurrentEdge == nil || len(a.Path) == 0 {
		return 0
	}
	
	// Time to complete current edge
	timeRemaining := (a.CurrentEdge.Length - a.CurrentPosition) / a.Velocity
	
	// Add time for remaining edges in path
	for i, edge := range a.Path {
		// Skip current edge as it's already calculated
		if i == 0 && edge == a.CurrentEdge {
			continue
		}
		
		// Time for this edge = length / velocity
		density := edge.Density()
		
		// Adjust velocity based on density
		// Using the non-linear car-following model from the paper
		adjustedVelocity := MaxVelocity
		if density > 0 {
			// Using the formula: v_i(t) = λ ln|x_i(t) - x_{i-1}(t)| + γ
			// where λ = v_max / ln(1/(ρ_max * L))
			// and γ = -λ ln|L|
			lambda := MaxVelocity / math.Log(1.0/(MaxDensity*VehicleLen))
			gamma := -lambda * math.Log(VehicleLen)
			
			// Approximate distance between vehicles
			distanceBetweenVehicles := 1.0 / density
			
			// Calculate velocity using car-following model
			adjustedVelocity = lambda*math.Log(distanceBetweenVehicles) + gamma
			
			// Ensure velocity is within limits
			if adjustedVelocity > MaxVelocity {
				adjustedVelocity = MaxVelocity
			}
			if adjustedVelocity < 1.0 {
				adjustedVelocity = 1.0 // Minimum velocity
			}
		}
		
		edgeTime := edge.Length / adjustedVelocity
		timeRemaining += edgeTime
	}
	
	return timeRemaining
}

// MoveAgent simulates the movement of an agent for a specific time step
func (g *Graph) MoveAgent(agent *Agent, timeStep float64) bool {
	if agent.CurrentEdge == nil || agent.Destination == nil {
		return false
	}
	
	// Calculate distance to move based on velocity and time step
	distanceToMove := agent.Velocity * timeStep
	
	// Check if agent will reach the end of the current edge
	remainingDistance := agent.CurrentEdge.Length - agent.CurrentPosition
	
	if distanceToMove >= remainingDistance {
		// Agent reaches the end of the current edge
		
		// Remove from current edge
		g.RemoveAgentFromEdge(agent, agent.CurrentEdge)
		
		// Check if destination reached
		if agent.CurrentEdge.To == agent.Destination {
			agent.CurrentEdge = nil
			return true // Destination reached
		}
		
		// Update path at intersection based on deviation rate
		if rand.Float64() < agent.DeviationRate {
			// Recalculate path with random deviation
			g.RecalculatePathWithDeviation(agent)
		} else if len(agent.Path) > 0 {
			// Remove current edge from path
			agent.Path = agent.Path[1:]
		}
		
		// Update envy matrix at intersection
		agent.UpdateEnvyMatrix(g)
		
		// Move to next edge if available
		if len(agent.Path) > 0 {
			nextEdge := agent.Path[0]
			
			// Check if next edge has capacity
			if len(nextEdge.Vehicles) < nextEdge.MaxCapacity {
				agent.CurrentEdge = nextEdge
				agent.CurrentPosition = 0
				nextEdge.Vehicles = append(nextEdge.Vehicles, agent)
				
				// Update current state
				g.CurrentState.EdgeVehicles[nextEdge.ID] = append(g.CurrentState.EdgeVehicles[nextEdge.ID], agent)
				
				// Calculate new time
				remainingTime := (distanceToMove - remainingDistance) / agent.Velocity
				if remainingTime > 0 {
					// Continue moving on new edge
					agent.CurrentPosition += remainingTime * agent.Velocity
				}
			} else {
				// Edge is at capacity, agent waits
				// For simplicity, place at beginning of edge but don't move
				agent.CurrentEdge = nextEdge
				agent.CurrentPosition = 0
				nextEdge.Vehicles = append(nextEdge.Vehicles, agent)
				
				// Update current state
				g.CurrentState.EdgeVehicles[nextEdge.ID] = append(g.CurrentState.EdgeVehicles[nextEdge.ID], agent)
			}
		} else {
			// No more edges in path
			return false
		}
	} else {
		// Agent still on the same edge
		agent.CurrentPosition += distanceToMove
	}
	
	// Update velocity based on car-following model
	g.UpdateAgentVelocity(agent)
	
	return false // Destination not reached yet
}

// RemoveAgentFromEdge removes an agent from an edge
func (g *Graph) RemoveAgentFromEdge(agent *Agent, edge *Edge) {
	// Remove from edge's vehicles
	for i, v := range edge.Vehicles {
		if v == agent {
			edge.Vehicles = append(edge.Vehicles[:i], edge.Vehicles[i+1:]...)
			break
		}
	}
	
	// Remove from current state
	vehicles := g.CurrentState.EdgeVehicles[edge.ID]
	for i, v := range vehicles {
		if v == agent {
			g.CurrentState.EdgeVehicles[edge.ID] = append(vehicles[:i], vehicles[i+1:]...)
			break
		}
	}
}

// UpdateAgentVelocity updates the velocity of an agent based on the car-following model
func (g *Graph) UpdateAgentVelocity(agent *Agent) {
	if agent.CurrentEdge == nil {
		return
	}
	
	// Find the agent in front, if any
	var frontAgent *Agent
	minDistance := Infinity
	
	for _, v := range agent.CurrentEdge.Vehicles {
		if v != agent && v.CurrentPosition > agent.CurrentPosition {
			distance := v.CurrentPosition - agent.CurrentPosition
			if distance < minDistance {
				minDistance = distance
				frontAgent = v
			}
		}
	}
	
	// Check if close to the end of the road for braking
	brakeZoneStart := agent.CurrentEdge.Length * (1.0 - BrakeRatio/100.0)
	
	if agent.CurrentPosition >= brakeZoneStart {
		// In braking zone - decelerate based on distance to end
		remainingDist := agent.CurrentEdge.Length - agent.CurrentPosition
		ratio := remainingDist / (agent.CurrentEdge.Length * BrakeRatio / 100.0)
		agent.Velocity = MaxVelocity * ratio
		if agent.Velocity < 1.0 {
			agent.Velocity = 1.0 // Minimum velocity
		}
	} else if frontAgent != nil {
		// Apply car-following model
		// Calculate lambda and gamma parameters
		lambda := MaxVelocity / math.Log(1.0/(MaxDensity*VehicleLen))
		gamma := -lambda * math.Log(VehicleLen)
		
		// Calculate distance to front vehicle
		distance := frontAgent.CurrentPosition - agent.CurrentPosition
		
		// Apply non-linear car-following model
		// v_i(t) = λ ln|x_i(t) - x_{i-1}(t)| + γ
		newVelocity := lambda*math.Log(distance) + gamma
		
		// Ensure velocity is within limits
		if newVelocity > MaxVelocity {
			newVelocity = MaxVelocity
		}
		if newVelocity < 1.0 {
			newVelocity = 1.0 // Minimum velocity
		}
		
		agent.Velocity = newVelocity
	} else {
		// No vehicle in front, drive at max velocity
		agent.Velocity = MaxVelocity
	}
}

// RecalculatePathWithDeviation recalculates the path with some deviation
func (g *Graph) RecalculatePathWithDeviation(agent *Agent) {
	if agent.CurrentEdge == nil || agent.CurrentEdge.To == nil {
		return
	}
	
	// Current node is the 'To' node of the current edge
	currentNode := agent.CurrentEdge.To
	
	// Get all outgoing edges from current node
	outEdges := g.GetOutgoingEdges(currentNode)
	
	if len(outEdges) == 0 {
		return
	}
	
	// Randomly select an outgoing edge that's different from the planned path
	var nextEdge *Edge
	
	// If there's only one outgoing edge, use it
	if len(outEdges) == 1 {
		nextEdge = outEdges[0]
	} else {
		// Try to pick an edge that's different from the originally planned one
		plannedNextEdge := agent.Path[0]
		availableEdges := make([]*Edge, 0)
		
		for _, edge := range outEdges {
			if edge != plannedNextEdge {
				availableEdges = append(availableEdges, edge)
			}
		}
		
		// If no alternative edges available, use the planned one
		if len(availableEdges) == 0 {
			nextEdge = plannedNextEdge
		} else {
			// Pick a random alternative edge
			nextEdge = availableEdges[rand.Intn(len(availableEdges))]
		}
	}
	
	// Calculate new path from the next edge to destination
	remainingPath := g.Dijkstra(nextEdge.To, agent.Destination)
	
	// Update agent's path with the new edge followed by the new path
	agent.Path = append([]*Edge{nextEdge}, remainingPath...)
}

// UpdateSystemState updates the system state after a time step
func (g *Graph) UpdateSystemState(timeStep float64) {
	// Increment system time
	g.Time += timeStep
	
	// Copy current state for history
	edgeVehiclesCopy := make(map[int][]*Agent)
	for id, vehicles := range g.CurrentState.EdgeVehicles {
		vehiclesCopy := make([]*Agent, len(vehicles))
		copy(vehiclesCopy, vehicles)
		edgeVehiclesCopy[id] = vehiclesCopy
	}
	
	// Save current state to history
	g.StateHistory = append(g.StateHistory, State{
		ID:           g.StateCounter,
		EdgeVehicles: edgeVehiclesCopy,
		Time:         g.Time,
	})
	
	g.StateCounter++
	
	// Update current state ID and time
	g.CurrentState.ID = g.StateCounter
	g.CurrentState.Time = g.Time
}

// SimulateTimeStep simulates system evolution for a time step
func (g *Graph) SimulateTimeStep(timeStep float64) {
	// Move all agents
	for _, agent := range g.Agents {
		if agent.CurrentEdge != nil {
			g.MoveAgent(agent, timeStep)
		}
	}
	
	// Update system state
	g.UpdateSystemState(timeStep)
}

// DisplaySystemState prints the current system state
func (g *Graph) DisplaySystemState() {
	fmt.Printf("System State at Time: %.2f\n", g.Time)
	fmt.Printf("State ID: %d\n", g.CurrentState.ID)
	
	for _, edge := range g.Edges {
		fmt.Printf("Edge %d (From %d to %d): Length=%.2f, Density=%.4f\n", 
			edge.ID, edge.From.ID, edge.To.ID, edge.Length, edge.Density())
		
		fmt.Printf("  Vehicles: [")
		for i, agent := range edge.Vehicles {
			if i > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("Agent %d (Pos: %.2f, Vel: %.2f)", agent.ID, agent.CurrentPosition, agent.Velocity)
		}
		fmt.Printf("]\n")
	}
	fmt.Println()
}

// CalculateEnvyFreenessMetric calculates a metric to quantify the envy-freeness of the system
func (g *Graph) CalculateEnvyFreenessMetric() float64 {
	if len(g.Agents) <= 1 {
		return 1.0 // Perfect envy-freeness with 0 or 1 agent
	}
	
	envySum := 0.0
	pairCount := 0
	
	// Compare each pair of agents
	for i, agent1 := range g.Agents {
		for j, agent2 := range g.Agents {
			if i == j {
				continue
			}
			
			// Calculate expected times
			time1 := agent1.CalculateExpectedTime()
			time2 := agent2.CalculateExpectedTime()
			
			// Skip if either agent has completed their journey
			if time1 == 0 || time2 == 0 {
				continue
			}
			
			// Calculate envy as the absolute difference in normalized costs
			// (Lower values mean less envy)
			envy := math.Abs((time1 / agent1.ExpectedTime) - (time2 / agent2.ExpectedTime))
			envySum += envy
			pairCount++
		}
	}
	
	if pairCount == 0 {
		return 1.0
	}
	
	// Return the envy-freeness score (1 - average envy)
	// A score of 1.0 means perfect envy-freeness, lower scores indicate more envy
	return 1.0 - (envySum / float64(pairCount))
}

// Main function to demonstrate the system
func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())
	
	// Create a new traffic network
	graph := NewGraph()
	
	// Create nodes (intersections)
	node1 := graph.AddNode(1, 0, 0)
	node2 := graph.AddNode(2, 100, 0)
	node3 := graph.AddNode(3, 200, 0)
	node4 := graph.AddNode(4, 300, 0)
	node5 := graph.AddNode(5, 100, 100)
	node6 := graph.AddNode(6, 200, 100)
	
	// Create edges (roads)
	graph.AddEdge(1, node1, node2, 100, 5)
	graph.AddEdge(2, node2, node3, 100, 5)
	graph.AddEdge(3, node3, node4, 100, 5)
	graph.AddEdge(4, node2, node5, 100, 5)
	graph.AddEdge(5, node5, node6, 100, 5)
	graph.AddEdge(6, node6, node4, 100, 5)
	
	// Add agents with different deviation rates
	graph.AddAgent(1, node1, node4, 0.0)  // Will not deviate
	graph.AddAgent(2, node1, node4, 0.5)  // 50% chance to deviate
	graph.AddAgent(3, node1, node4, 0.8)  // 80% chance to deviate
	
	fmt.Println("Initial system state:")
	graph.DisplaySystemState()
	
	// Simulate for several time steps
	numSteps := 200
	timeStep := 0.5
	
	for step := 1; step <= numSteps; step++ {
		// Simulate one time step
		graph.SimulateTimeStep(timeStep)
		
		// Display system state periodically
		if step%20 == 0 {
			fmt.Printf("\n=== Time Step %d ===\n", step)
			graph.DisplaySystemState()
			
			// Calculate and display envy-freeness metric
			envyMetric := graph.CalculateEnvyFreenessMetric()
			fmt.Printf("Envy-Freeness Metric: %.4f (1.0 is perfect envy-freeness)\n", envyMetric)
			
			// Display path information for each agent
			for _, agent := range graph.Agents {
				if agent.CurrentEdge == nil {
					fmt.Printf("Agent %d has reached destination\n", agent.ID)
				} else {
					fmt.Printf("Agent %d is on edge %d, ", agent.ID, agent.CurrentEdge.ID)
					fmt.Printf("Remaining path: [")
					for i, edge := range agent.Path {
						if i > 0 {
							fmt.Printf(" -> ")
						}
						fmt.Printf("%d", edge.ID)
					}
					fmt.Printf("]\n")
					
					// Print last row of envy matrix
					if len(agent.EnvyMatrix) > 0 {
						lastRow := agent.EnvyMatrix[len(agent.EnvyMatrix)-1]
						fmt.Printf("  Envy ratios: [")
						for i, ratio := range lastRow {
							if i > 0 {
								fmt.Printf(", ")
							}
							fmt.Printf("%.2f", ratio)
						}
						fmt.Printf("]\n")
					}
				}
			}
		}
		
		// Check if all agents have reached their destinations
		allDone := true
		for _, agent := range graph.Agents {
			if agent.CurrentEdge != nil {
				allDone = false
				break
			}
		}
		
		if allDone {
			fmt.Println("\nAll agents have reached their destinations.")
			break
		}
	}
	
	// Print system state history summary
	fmt.Println("\n=== System State History Summary ===")
	fmt.Printf("Total number of states: %d\n", len(graph.StateHistory))
	fmt.Printf("Initial state time: %.2f\n", graph.StateHistory[0].Time)
	fmt.Printf("Final state time: %.2f\n", graph.StateHistory[len(graph.StateHistory)-1].Time)
}
