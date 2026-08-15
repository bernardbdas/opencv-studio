/**
 * Utility to convert raw LaTeX expressions to clean, human-readable Unicode mathematical text.
 */
export function formatMathText(text: string): string {
  if (!text) return text;
  let formatted = text;

  // Clean LaTeX inline / block math delimiters
  formatted = formatted.replace(/\\\(|\\\)/g, '');
  formatted = formatted.replace(/\$\$/g, '');
  formatted = formatted.replace(/\$/g, '');

  // Subscripts & Superscripts
  formatted = formatted.replace(/_i\b/g, 'ᵢ');
  formatted = formatted.replace(/_x\b/g, 'ₓ');
  formatted = formatted.replace(/_y\b/g, 'ᵧ');
  formatted = formatted.replace(/_z\b/g, 'z');
  formatted = formatted.replace(/_g\b/g, 'g');
  formatted = formatted.replace(/_1\b/g, '₁');
  formatted = formatted.replace(/_2\b/g, '₂');
  formatted = formatted.replace(/\^2\b/g, '²');
  formatted = formatted.replace(/\^T\b/g, 'ᵀ');

  // Specific multi-character sub/superscripts
  formatted = formatted.replace(/_\{([a-zA-Z0-9]+)\}/g, (_, sub) => {
    // Map standard sub words to flat text or keep it clean
    return `_${sub}`;
  });

  // Greek letters
  formatted = formatted.replace(/\\alpha/g, 'α');
  formatted = formatted.replace(/\\beta/g, 'β');
  formatted = formatted.replace(/\\sigma/g, 'σ');
  formatted = formatted.replace(/\\delta/g, 'δ');
  formatted = formatted.replace(/\\psi/g, 'ψ');
  formatted = formatted.replace(/\\theta/g, 'θ');
  formatted = formatted.replace(/\\lambda/g, 'λ');

  // Math Operators & Accents
  formatted = formatted.replace(/\\in\b/g, '∈');
  formatted = formatted.replace(/\\cdot/g, '·');
  formatted = formatted.replace(/\\quad/g, '   ');
  formatted = formatted.replace(/\\qquad/g, '      ');
  formatted = formatted.replace(/\\left\(/g, '(');
  formatted = formatted.replace(/\\right\)/g, ')');
  formatted = formatted.replace(/\\left\[/g, '[');
  formatted = formatted.replace(/\\right\]/g, ']');
  formatted = formatted.replace(/\\left\\{/g, '{');
  formatted = formatted.replace(/\\right\\}/g, '}');
  formatted = formatted.replace(/\\left/g, '');
  formatted = formatted.replace(/\\right/g, '');
  
  // Logical / Comparison operators
  formatted = formatted.replace(/\\le/g, '≤');
  formatted = formatted.replace(/\\ge/g, '≥');
  formatted = formatted.replace(/\\land/g, '∧');
  formatted = formatted.replace(/\\min/g, 'min');
  formatted = formatted.replace(/\\max/g, 'max');

  // Common LaTeX formulas
  formatted = formatted.replace(/\\sqrt\{([^\}]+)\}/g, '√($1)');
  formatted = formatted.replace(/\\frac\{([^\}]+)\}\{([^\}]+)\}/g, '($1 / $2)');
  formatted = formatted.replace(/\\arctan/g, 'arctan');
  formatted = formatted.replace(/\\lceil/g, '⌈');
  formatted = formatted.replace(/\\rceil/g, '⌉');
  formatted = formatted.replace(/\\vec\{([^\}]+)\}/g, '$1⃗');

  // Strip remaining \text{word} wrappers
  formatted = formatted.replace(/\\text\s*\{\s*([a-zA-Z0-9_\-\s]+)\s*\}/g, '$1');

  return formatted;
}
