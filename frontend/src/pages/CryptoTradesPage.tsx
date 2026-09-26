import TradesPage from './TradesPage'

export default function CryptoTradesPage() {
  return (
    <TradesPage
      endpointBase="/crypto-signals"
      title="CRYPTO"
      subtitle="24/7 crypto scanner — BTC/ETH regime, funding-aware scoring, top-N by market cap"
      lookupPlaceholder="Enter ticker (e.g., BTC, ETH, SOL)"
    />
  )
}
