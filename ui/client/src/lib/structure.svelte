<script>
  import * as mol3 from '3dmol'
  import Chart from './chart.svelte'
  import { selectedNode, fetchStructures } from './data'
  import Loading from './loading.svelte'

  let loading = false
  let loadedNode = null
  let catalysts = []
  let catalystIndex = 0
  let pathways = []
  let pathwayIndex = 0
  let ref = null
  let ids = []
  let values = []
  let viewer = null
  let structures = []
  let structureIndex = 0

  $: if (ref && viewer) {
    new ResizeObserver(() => {
      viewer.resize()
      viewer.zoomTo()
      viewer.render()
    }).observe(ref)
  }

  $: if (!viewer && structures.length && ref) {
    viewer = mol3.createViewer(ref, {
      // backgroundAlpha: 0,
      // This is a hack because backgroundAlpha does not work in Safari (or in newer
      // versions of 3dmol). The background color is set to match that of the parent,
      // which is bg-gray-950.
      backgroundColor: 'rgb(3, 7, 18)',
    })
    viewer.getCanvas().style.position = 'static'
  }

  $: if (viewer && structures.length) {
    viewer.removeAllModels()
    if (structures[structureIndex]) {
      viewer.addModel(structures[structureIndex].structure, 'xyz')
      viewer.addModel(structures[structureIndex].structure, 'xyz')
      viewer.setStyle(
        {},
        {
          stick: {
            // https://3dmol.csb.pitt.edu/doc/colors.ts.html
            colorscheme: 'Jmol',
          },
          sphere: {
            scale: 0.3,
            colorscheme: 'Jmol',
          },
        }
      )
      viewer.zoomTo()
      viewer.render()
    }
  }

  $: if ($selectedNode && $selectedNode !== loadedNode) {
    reset()
    loading = true
    fetchStructures($selectedNode.id)
      .then((d) => {
        pathways = d.pathways
        catalysts = d.catalysts
      })
      .finally(() => {
        loading = false
        loadedNode = $selectedNode
      })
  }

  $: if (!$selectedNode) {
    reset()
  }

  $: if (pathways.length) {
    structures = pathways[pathwayIndex].structures[catalystIndex]
  }

  $: if (structures.length) {
    ids = structures.map((t) => (t ? t.id : null))
    values = structures.map((t, i) => ({
      x: i,
      y: t ? t.energy : 0,
    }))
  }

  function reset() {
    if (viewer) {
      viewer.removeAllModels()
      viewer.render()
    }
    catalysts = []
    catalystIndex = 0
    pathways = []
    pathwayIndex = 0
    structures = []
    structureIndex = 0
    loadedNode = null
  }
</script>

<div class="relative h-full w-full">
  <div class="h-full w-full" bind:this={ref}></div>
  {#if pathways.length && catalysts.length}
    <div class="absolute left-2 right-2 top-2 flex-col">
      <div class="items center flex justify-between">
        <div class="flex items-center">
          <span class="text-gray-500">Pathway:</span>
          <select class="bg-transparent font-mono" bind:value={pathwayIndex}>
            {#each pathways as p, i}
              <option value={i} selected={pathwayIndex === i}>
                {p.reactants.join(' ')}
              </option>
            {/each}
          </select>
        </div>
        <div class="flex items-center">
          <span class="text-gray-500">Catalyst:</span>
          <select class="bg-transparent font-mono" bind:value={catalystIndex}>
            {#each catalysts as c, i}
              <option value={i} selected={catalystIndex === i}>
                {c.join(' ')}
              </option>
            {/each}
          </select>
        </div>
      </div>
    </div>
  {/if}
  {#if structures.length && values.length && !loading}
    <div class="absolute bottom-2 left-2 right-2 flex-row">
      <div class="flex items-end justify-between">
        <div class="flex flex-col items-center">
          <div class="h-12 w-full">
            <Chart data={values} markerIndex={structureIndex} xPadding={8} />
          </div>
          <input
            class="w-full"
            type="range"
            min="0"
            max={structures.length - 1}
            bind:value={structureIndex}
          />
        </div>
        <div class="flex items-center gap-2">
          <span class="font-mono text-yellow-500">
            {pathways[pathwayIndex].reactants[structureIndex]}
            +
            {catalysts[catalystIndex].join(' ')}
          </span>
          <span class="text-gray-500">//</span>
          <span>Energy:</span>
          <span class="font-mono text-yellow-500">
            {#if structures[structureIndex]}
              {structures[structureIndex].energy.toFixed(2)} eV
            {:else}
              &mdash;
            {/if}
          </span>
        </div>
      </div>
    </div>
  {:else}
    <div
      class="absolute bottom-0 left-0 right-0 top-0 flex h-full w-full items-center
        justify-center p-16 text-center text-lg text-gray-500"
    >
      {#if loading}
        <Loading size="lg" />
      {:else if $selectedNode}
        No reactions available to visualize.
      {:else}
        Select a node to view reactions and associated energy values.
      {/if}
    </div>
  {/if}
</div>
