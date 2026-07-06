import React from "react";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  glow?: boolean;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = "",
  glow = false,
}) => {
  return (
    <div
      className={`liquid-glass rounded-2xl p-5 transition-all duration-300 hover:border-white/15 ${
        glow ? "liquid-glass-glow" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
};
