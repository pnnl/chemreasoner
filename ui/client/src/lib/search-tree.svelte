<script>
  import { onMount } from 'svelte'
  import { fade } from 'svelte/transition'

  import { fetchSearchData, nodes as nodesStore, selectedNode } from './data'
  import Graph from './graph.svelte'
  import { traverse } from './util'
  import Loading from './loading.svelte'

  let loading = false
  let nodes = []
  let edges = []
  let style
  let cy

  let minReward = 0
  let maxReward = 0
  let minDepth = 0
  let maxDepth = 0

  const blue = 'rgb(7, 89, 133)'

  $: layout = {
    // name: 'dagre',
    // rankDir: 'TB',
    // ranker: 'tight-tree',
    // name: 'concentric',
    // concentric: (node) => {
    //   let n = 0
    //   pathToRoot(node, () => n++)
    //   return n
    // },
    name: 'cose',
    randomize: false,
    animate: false,
    fit: true,
    padding: 10,
  }

  $: style = [
    {
      selector: 'node',
      style: {
        width: `mapData(reward, ${minReward}, ${maxReward}, 15, 60)`,
        height: `mapData(reward, ${minReward}, ${maxReward}, 15, 60)`,
        opacity: `mapData(depth, ${minDepth}, ${maxDepth}, 0.1, 1)`,
        'font-size': '18',
        'font-weight': 'bold',
        content: `data(label)`,
        'text-valign': 'center',
        'text-wrap': 'wrap',
        'text-max-width': '140',
        'background-color': '#aaa',
        'border-color': 'white',
        'border-width': '3',
        color: '#0e76ba',
      },
    },
    // {
    //   selector: 'node[root = "true"]',
    //   style: {
    //     'background-color': 'rgb(7, 89, 133)',
    //     opacity: 0.5,
    //   },
    // },
    // {
    //   selector: 'node[leaf = "true"]',
    //   style: {
    //     'background-color': 'rgb(253, 224, 71)',
    //     opacity: 1,
    //   },
    // },
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        color: '#aaa',
        'text-background-color': '#ffffff',
        'text-background-opacity': '1',
        'text-background-padding': '3',
        width: '3',
        // 'target-arrow-shape': 'triangle',
        'line-color': '#444',
        // 'target-arrow-color': '#444',
        'font-weight': 'bold',
      },
    },
    {
      selector: 'edge[label]',
      style: {
        content: `data(label)`,
      },
    },
    {
      selector: 'edge.label',
      style: {
        'line-color': 'white',
        'target-arrow-color': 'white',
      },
    },
    {
      selector: 'node.diminished',
      style: {
        opacity: 0.1,
      },
    },
    {
      selector: 'edge.diminished',
      style: {
        opacity: 0.1,
      },
    },
    {
      selector: 'node:selected',
      style: {
        'background-color': blue,
        color: '#0e76ba',
        opacity: 1,
        'border-color': 'white',
        'line-color': '#0e76ba',
        'target-arrow-color': '#0e76ba',
      },
    },
    {
      selector: 'edge.path',
      style: {
        'line-color': blue,
        'target-arrow-color': blue,
        opacity: 1,
      },
    },
    {
      selector: 'node.path',
      style: {
        'border-color': blue,
        opacity: 1,
      },
    },
  ]

  onMount(async () => {
    loading = true
    const tree = await fetchSearchData()
    loading = false
    traverse(tree, (node, depth) => {
      nodes.push({
        id: node.id,
        root: `${node.id === 0}`,
        leaf: `${!Boolean(node.children)}`,
        label: '',
        reward: Math.max(node.node_rewards, -1),
        depth,
        ...node.info,
      })

      if (node.children) {
        node.children.forEach((child) => {
          edges.push({
            id: `${node.id}-${child.id}`,
            source: node.id,
            target: child.id,
          })
        })
      }
    })
    ;[minReward, maxReward] = nodes.reduce(
      (acc, node) => {
        return [Math.min(acc[0], node.reward), Math.max(acc[1], node.reward)]
      },
      [Infinity, -Infinity]
    )
    ;[minDepth, maxDepth] = nodes.reduce(
      (acc, node) => {
        return [Math.min(acc[0], node.depth), Math.max(acc[1], node.depth)]
      },
      [Infinity, -Infinity]
    )
    ;[nodes, edges] = [nodes, edges]
    $nodesStore = nodes
  })

  $: if (cy) {
    cy.on('select', 'node', (e) => {
      $selectedNode = e.target.data()
      cy.elements().addClass('diminished')
      pathToRoot(e.target, (edge) => {
        edge.addClass('path')
        edge.source().addClass('path')
      })
    })
    cy.on('unselect', 'node', (e) => {
      $selectedNode = null
      cy.elements().removeClass('diminished path')
    })
  }

  function pathToRoot(node, fn) {
    let currentNode = node
    while (currentNode) {
      let incomingEdges = currentNode.incomers('edge')
      if (incomingEdges.length) {
        fn(incomingEdges[0])
        currentNode = incomingEdges[0].source()
      } else {
        currentNode = null
      }
    }
  }
</script>

{#if loading}
  <Loading />
{/if}

<Graph {nodes} {edges} {style} {layout} bind:cy />
