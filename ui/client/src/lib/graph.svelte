<script>
  import { onMount, onDestroy } from 'svelte'
  import cytoscape from 'cytoscape'
  import dagre from 'cytoscape-dagre'
  import spread from 'cytoscape-spread'
  import fcose from 'cytoscape-fcose'
  import cise from 'cytoscape-cise'

  export let nodes = []
  export let edges = []
  export let style = {}
  export let layout = {}
  export let cy = null

  let ref = null
  let ready = false

  onMount(() => {
    cytoscape.use(dagre)
    cytoscape.use(spread)
    cytoscape.use(fcose)
    cytoscape.use(cise)
  })

  onDestroy(() => {
    if (cy) {
      cy.destroy()
      cy = null
      ready = false
    }
  })

  $: if (ref) {
    cy = cytoscape({
      container: ref,
      style,
    })

    cy.ready(() => (ready = true))
  }

  $: if (ref) {
    new ResizeObserver(() => {
      if (cy) {
        cy.resize()
        cy.fit()
      }
    }).observe(ref)
  }

  $: if (cy && ready && nodes.length && edges.length && style) {
    nodes.forEach((n) =>
      cy.add({
        group: 'nodes',
        id: n.id,
        style: n.style,
        data: { ...n },
      })
    )
    edges.forEach((e) =>
      cy.add({
        group: 'edges',
        id: e.id,
        style: e.style,
        data: { ...e },
      })
    )
    cy.layout(layout).run()
  }
</script>

<div class="h-full w-full" bind:this={ref}></div>
