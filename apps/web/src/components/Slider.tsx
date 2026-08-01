interface SliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (val: number) => void;
}

export default function Slider({ label, min, max, step, value, onChange }: SliderProps) {
  return (
    <label style={{ display: 'block', marginBottom: '1rem', fontSize: '0.9rem', fontWeight: '500' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
        <span style={{ color: 'var(--text-main)' }}>{label}</span>
        <span style={{ color: 'var(--color-accent)', fontWeight: '700' }}>{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ 
          width: '100%', 
          cursor: 'pointer',
          accentColor: 'var(--color-accent)'
        }}
      />
    </label>
  );
}