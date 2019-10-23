
# Topological Sort

# You are given a list of arbitrary jobs that need to be completed; these jobs are represented by integers. You are also given a list of dependencies. A dependency is represented as a pair of jobs where the first job is prerequisite of the second one. In other words, the second job depends on the first one; it can only be completed once the first job is completed. Write a function that takes in a list of jobs and a list of dependencies and returns a list containing a valid order in which the given jobs can be completed. If no such order exists, the function should return an empty list.

# Sample input: [1, 2, 3, 4], [[1, 2], [1, 3], [3, 2], [4, 2], [4, 3]]
# Sample output: [1, 4, 3, 2] or [4, 1, 3, 2]


# O(j + d) time | O(j + d) space

def topologicalSort(jobs, deps):
    jobGraph = createJobGraph(jobs, deps)
    return getOrderedJobs(jobGraph)

def createJobGraph(jobs, deps):
    ## Add all JobNodes This merely adds vertices to the graph 
    graph = JobGraph(jobs)
    ##  Add prereqs (edges of the graph!)
    for prereq, job in deps:
        graph.addPrereq(job, prereq)
    return graph

def getOrderedJobs(graph):
    orderedJobs = []
    nodes = graph.nodes
    ### nodes is a list of all JobNodes to be sorted
    while len(nodes):
        node = nodes.pop()
        containsCycle = depthFirstTraverse(node, orderedJobs)
        if containsCycle:
            return []
    return orderedJobs

def depthFirstTraverse(node, orderedJobs):
    ## if node has been visited before return False! 
    if node.visited:
        return False
    ## if returning to a node that is in progress(visiting state) it means we found a cycle! 
    if node.visiting:
        return True
    
    ## otherwise start DFS 
    ## First Mark the Node as in progress/visiting
    node.visiting = True
    ## iterate over prereqs
    ## if we started by a node with no prereqs this fo loop will be skipped
    ## and this job will be added to the orderedJobs
    for prereqNode in node.prereqs:
        ## this recursive call will continue DFS until you reach a node with no prereqs then backtrack and continue DFS
        containsCycle = depthFirstTraverse(prereqNode, orderedJobs)
        if containsCycle:
            return True
    ### now that you visited all prereqs you can safely mark node as "visited" and annul its visiting state
    node.visited = True
    node.visiting = False
    orderedJobs.append(node.job)
    return False

class JobGraph:
    def __init__(self, jobs):
        ## will keep track of all nodes in a list of JobNodes (see class below)
        self.nodes = []
        ## In order to check membership we quickly build this hashtable where job number is key
        ## and value is the corresponding JobNode 
        
        self.graph = {}
        for job in jobs:
            self.addNode(job)
        ## this will create graph nodes and put them in the dictionary self.graph and the list self.nodes
        ## Note that as this point we did not create dependencies (edges of the graph!)

    def addPrereq(self, job, prereq):
        jobNode = self.getNode(job)
        prereqNode = self.getNode(prereq)
        jobNode.prereqs.append(prereqNode)

    def addNode(self, job):
        self.graph[job] = JobNode(job)
        self.nodes.append(self.graph[job])

    def getNode(self, job):
        if job not in self.graph:
            self.addNode(job)
        return self.graph[job]

class JobNode:
    def __init__(self, job):
        self.job = job
        self.prereqs = []
        self.visited = False
        self.visiting = False


jobs= [1, 2, 3, 4] 
deps=[[1, 2], [1, 3], [3, 2], [4, 2], [4, 3]]
topologicalSort(jobs,deps)

## 1 preq >> dep 2  , 1 preq >> dep 3 , 3 preq >> dep 2 , 4 preq >> dep 2  , 4 preq >> dep 3


#################################
# Solution 2 
###############################


# O(j + d) time | O(j + d) space
def topologicalSort(jobs, deps):
    jobGraph = createJobGraph(jobs, deps)
    return getOrderedJobs(jobGraph)

def createJobGraph(jobs, deps):
    graph = JobGraph(jobs)
    for job, dep in deps:
        graph.addDep(job, dep)
    return graph

def getOrderedJobs(graph):
    orderedJobs = []
    nodesWithNoPrereqs = list(filter(lambda node: node.numOfPrereqs == 0, graph.nodes))
    ## build a list with no prereqs
    ## iterate/traverse graphs until this list is []
    while len(nodesWithNoPrereqs):
        node = nodesWithNoPrereqs.pop()
        orderedJobs.append(node.job)
        ## update dependecy prereq count after removing this node from the graph! 
        removeDeps(node, nodesWithNoPrereqs)
    graphHasEdges = any(node.numOfPrereqs for node in graph.nodes)
    return [] if graphHasEdges else orderedJobs

def removeDeps(node, nodesWithNoPrereqs):
    ## iterate over the dependencies of the node that does not have prereqs
    ## update their prereq count by -1. if one or more end up with 0 prereqs add it to the nodes with no prereqs list!
    while len(node.deps):
        dep = node.deps.pop()
        dep.numOfPrereqs -= 1
        if dep.numOfPrereqs == 0:
            nodesWithNoPrereqs.append(dep)

class JobGraph:
    def __init__(self, jobs):
        self.nodes = []
        self.graph = {}
        for job in jobs:
            self.addNode(job)

    def addDep(self, job, dep):
        jobNode = self.getNode(job)
        depNode = self.getNode(dep)
        jobNode.deps.append(depNode)
        ## for a given jobnode every time you append a dependency need to increase numofprereqs of the
        ## dependency by 1 
        depNode.numOfPrereqs += 1

    def addNode(self, job):
        self.graph[job] = JobNode(job)
        self.nodes.append(self.graph[job])

    def getNode(self, job):
        if job not in self.graph:
            self.addNode(job)
        return self.graph[job]

class JobNode:
    def __init__(self, job):
        self.job = job
        self.deps = []  ## keep track of dependencies for a given jobnode (i.e. those nodes that need to wait for node to run!)
        self.numOfPrereqs = 0 
        
########
topologicalSort(jobs,deps)
## output [4, 1, 3, 2]
x_l=[0,1,2,5,5,9,0,4,2,0]
list(filter(lambda elem: elem == 0, x_l))
## Answer [0, 0, 0]

        
        
