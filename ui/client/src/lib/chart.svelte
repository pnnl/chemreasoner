<script>
  import { scaleLinear } from 'd3-scale'
  import { line, curveCatmullRom } from 'd3-shape'

  export let data = []
  export let color = 'rgba(255, 255, 255, 0.5)'
  export let markerIndex = null
  export let markerColor = '#eab308'
  export let markerWidth = 6
  export let xPadding = 0
  export let yPadding = 3

  let width
  let height
  let pathData = ''

  $: xDomain = data.map((d) => d.x)
  $: yDomain = data.map((d) => d.y)

  $: xScale = scaleLinear()
    .domain([Math.min.apply(null, xDomain), Math.max.apply(null, xDomain)])
    .range([0, width - 2 * xPadding])
  $: yScale = scaleLinear()
    .domain([Math.min.apply(null, yDomain), Math.max.apply(null, yDomain)])
    .range([0, height - 2 * yPadding])

  const lineGenerator = line()
    .x((d) => xScale(d.x) + xPadding)
    .y((d) => height - yScale(d.y) - yPadding)
    .curve(curveCatmullRom)

  $: if (width && height) {
    pathData = lineGenerator(data)
  }
</script>

<div class="h-full w-full" bind:clientWidth={width} bind:clientHeight={height}>
  {#if width > 0}
    <svg {width} {height}>
      <path
        d={pathData}
        style={`stroke: ${color}; stroke-width: 1; fill: none`}
      />
      {#if markerIndex !== null}
        <rect
          x={xScale(data[markerIndex || 0].x) - markerWidth / 2 + xPadding}
          y={height - yScale(data[markerIndex || 0].y)}
          width={markerWidth}
          height={yScale(data[markerIndex || 0].y)}
          fill={markerColor}
          rx={markerWidth / 2}
        />
      {/if}
    </svg>
  {/if}
</div>
