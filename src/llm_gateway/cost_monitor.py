"""
Advanced cost monitoring and budget management system for LLM gateway
with predictive analytics, alerts, and cost optimization recommendations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import numpy as np

from .gateway import ModelProvider, RequestMetrics, CostLimit, CostLimitType
from ..database.connection import get_db_session
from ..utils.logger import get_logger

logger = get_logger(__name__)

class AlertLevel(Enum):
    """Cost alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class BudgetStatus(Enum):
    """Budget status indicators."""
    HEALTHY = "healthy"         # < 50% of budget used
    MODERATE = "moderate"       # 50-75% of budget used
    HIGH = "high"              # 75-90% of budget used
    CRITICAL = "critical"      # 90-100% of budget used
    EXCEEDED = "exceeded"      # > 100% of budget used

@dataclass
class CostAlert:
    """Cost alert notification."""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    limit_type: CostLimitType
    current_usage: float
    limit_amount: float
    recommended_actions: List[str] = field(default_factory=list)

@dataclass
class BudgetForecast:
    """Budget forecast and projections."""
    period_type: CostLimitType
    current_usage: float
    budget_limit: float
    projected_usage: float
    projected_end_date: datetime
    confidence_score: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    days_until_exhaustion: Optional[int] = None

@dataclass
class CostAnalysis:
    """Comprehensive cost analysis."""
    analysis_period: timedelta
    total_cost: float
    cost_per_hour: float
    cost_per_request: float
    most_expensive_model: str
    most_used_model: str
    cost_efficiency_score: float
    optimization_opportunities: List[Dict[str, Any]]

class CostMonitor:
    """
    Advanced cost monitoring system with predictive analytics,
    budget management, and automated optimization recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = get_db_session()
        
        # Alert system
        self.alert_handlers: List[Callable] = []
        self.alert_history: List[CostAlert] = []
        
        # Budget tracking
        self.budget_limits: Dict[CostLimitType, float] = {}
        self.cost_forecasts: Dict[CostLimitType, BudgetForecast] = {}
        
        # Cost optimization
        self.optimization_rules: List[Dict[str, Any]] = []
        self.cost_patterns: Dict[str, Any] = {}
        
        # Analytics
        self.usage_trends: Dict[str, List[float]] = defaultdict(list)
        self._monitoring = False
        
    async def start_monitoring(self):
        """Start continuous cost monitoring."""
        
        if self._monitoring:
            return
        
        self._monitoring = True
        
        # Start background monitoring tasks
        asyncio.create_task(self._continuous_monitoring())
        asyncio.create_task(self._periodic_analysis())
        asyncio.create_task(self._forecast_update())
        
        logger.info("Cost monitoring started")
    
    async def stop_monitoring(self):
        """Stop cost monitoring."""
        self._monitoring = False
        logger.info("Cost monitoring stopped")
    
    async def _continuous_monitoring(self):
        """Continuous monitoring loop for real-time alerts."""
        
        while self._monitoring:
            try:
                await self._check_budget_thresholds()
                await self._check_spending_velocity()
                await self._check_unusual_patterns()
                
                # Monitor every 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Cost monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Longer wait on error
    
    async def _periodic_analysis(self):
        """Periodic deep analysis and optimization recommendations."""
        
        while self._monitoring:
            try:
                await self._analyze_cost_trends()
                await self._generate_optimization_recommendations()
                await self._update_forecasts()
                
                # Deep analysis every 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Periodic analysis error: {str(e)}")
                await asyncio.sleep(300)  # Retry after 5 minutes
    
    async def _forecast_update(self):
        """Update cost forecasts periodically."""
        
        while self._monitoring:
            try:
                await self._update_all_forecasts()
                
                # Update forecasts every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Forecast update error: {str(e)}")
                await asyncio.sleep(1800)  # Retry after 30 minutes
    
    async def set_budget_limit(self, period: CostLimitType, amount_usd: float):
        """Set budget limit for specified period."""
        
        self.budget_limits[period] = amount_usd
        logger.info(f"Set {period.value} budget limit: ${amount_usd:.2f}")
        
        # Generate initial forecast
        await self._generate_forecast(period)
    
    async def add_cost_record(self, metrics: RequestMetrics):
        """Add cost record and trigger monitoring checks."""
        
        try:
            # Update usage trends
            hour_key = metrics.timestamp.strftime("%Y%m%d_%H")
            self.usage_trends[hour_key].append(metrics.cost_usd)
            
            # Keep only recent trends (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            cutoff_key = cutoff_time.strftime("%Y%m%d_%H")
            
            # Remove old trend data
            old_keys = [k for k in self.usage_trends.keys() if k < cutoff_key]
            for key in old_keys:
                del self.usage_trends[key]
            
            # Check for immediate threshold violations
            await self._check_immediate_thresholds(metrics)
            
        except Exception as e:
            logger.error(f"Error adding cost record: {str(e)}")
    
    async def _check_budget_thresholds(self):
        """Check budget threshold violations."""
        
        for period, budget_limit in self.budget_limits.items():
            current_usage = await self._get_current_usage(period)
            utilization = (current_usage / budget_limit) * 100
            
            # Generate alerts based on utilization
            if utilization >= 100:
                await self._create_alert(
                    AlertLevel.EMERGENCY,
                    f"{period.value.title()} budget exceeded: ${current_usage:.4f} / ${budget_limit:.2f}",
                    period,
                    current_usage,
                    budget_limit,
                    ["Immediately review spending", "Consider pausing non-critical requests"]
                )
            elif utilization >= 90:
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"{period.value.title()} budget critical: {utilization:.1f}% used",
                    period,
                    current_usage,
                    budget_limit,
                    ["Review upcoming requests", "Enable aggressive cost optimization"]
                )
            elif utilization >= 75:
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"{period.value.title()} budget high: {utilization:.1f}% used",
                    period,
                    current_usage,
                    budget_limit,
                    ["Monitor spending closely", "Consider using cheaper models"]
                )
    
    async def _check_spending_velocity(self):
        """Check if spending rate is unusually high."""
        
        try:
            # Calculate spending rate for last hour
            current_hour_costs = []
            current_time = datetime.now()
            
            hour_key = current_time.strftime("%Y%m%d_%H")
            if hour_key in self.usage_trends:
                current_hour_costs = self.usage_trends[hour_key]
            
            if len(current_hour_costs) < 2:
                return  # Not enough data
            
            current_hour_total = sum(current_hour_costs)
            
            # Compare with average of previous 24 hours
            previous_hours_avg = await self._get_average_hourly_spending(24)
            
            if previous_hours_avg > 0 and current_hour_total > previous_hours_avg * 3:
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"Unusual spending spike: ${current_hour_total:.4f} (3x average)",
                    CostLimitType.HOURLY,
                    current_hour_total,
                    previous_hours_avg * 3,
                    ["Investigate unusual activity", "Check for runaway processes"]
                )
            
        except Exception as e:
            logger.error(f"Spending velocity check failed: {str(e)}")
    
    async def _check_unusual_patterns(self):
        """Detect unusual spending patterns."""
        
        try:
            # Get recent usage data
            recent_usage = await self._get_usage_data(hours=6)
            
            if len(recent_usage) < 10:
                return  # Not enough data
            
            # Check for unusual model usage
            model_usage = defaultdict(float)
            for metrics in recent_usage:
                model_usage[metrics.model_name] += metrics.cost_usd
            
            total_cost = sum(model_usage.values())
            
            for model_name, cost in model_usage.items():
                usage_percent = (cost / total_cost) * 100
                
                # Alert if single model accounts for > 80% of costs
                if usage_percent > 80:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        f"Model {model_name} dominates spending: {usage_percent:.1f}% of costs",
                        CostLimitType.HOURLY,
                        cost,
                        total_cost,
                        ["Review model selection strategy", "Check for automated processes"]
                    )
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
    
    async def _get_current_usage(self, period: CostLimitType) -> float:
        """Get current usage for specified period."""
        
        now = datetime.now()
        
        if period == CostLimitType.HOURLY:
            start_time = now.replace(minute=0, second=0, microsecond=0)
        elif period == CostLimitType.DAILY:
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == CostLimitType.WEEKLY:
            days_back = now.weekday()
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
        elif period == CostLimitType.MONTHLY:
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # TOTAL
            start_time = datetime.min
        
        usage_data = await self._get_usage_data_since(start_time)
        return sum(metrics.cost_usd for metrics in usage_data)
    
    async def _get_usage_data(self, hours: int) -> List[RequestMetrics]:
        """Get usage data for specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return await self._get_usage_data_since(cutoff_time)
    
    async def _get_usage_data_since(self, start_time: datetime) -> List[RequestMetrics]:
        """Get usage data since specified time."""
        
        # This would query the database for actual metrics
        # For now, return empty list as placeholder
        return []
    
    async def _get_average_hourly_spending(self, hours: int) -> float:
        """Calculate average hourly spending over specified hours."""
        
        usage_data = await self._get_usage_data(hours)
        
        if not usage_data:
            return 0.0
        
        # Group by hour
        hourly_costs = defaultdict(float)
        
        for metrics in usage_data:
            hour_key = metrics.timestamp.strftime("%Y%m%d_%H")
            hourly_costs[hour_key] += metrics.cost_usd
        
        if not hourly_costs:
            return 0.0
        
        return statistics.mean(hourly_costs.values())
    
    async def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        limit_type: CostLimitType,
        current_usage: float,
        limit_amount: float,
        recommended_actions: List[str]
    ):
        """Create and dispatch cost alert."""
        
        alert = CostAlert(
            alert_id=f"alert_{int(datetime.now().timestamp())}",
            level=level,
            message=message,
            timestamp=datetime.now(),
            limit_type=limit_type,
            current_usage=current_usage,
            limit_amount=limit_amount,
            recommended_actions=recommended_actions
        )
        
        # Store alert
        self.alert_history.append(alert)
        
        # Keep only recent alerts (last 30 days)
        cutoff_time = datetime.now() - timedelta(days=30)
        self.alert_history = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        # Dispatch to handlers
        await self._dispatch_alert(alert)
        
        logger.warning(f"Cost alert [{level.value.upper()}]: {message}")
    
    async def _dispatch_alert(self, alert: CostAlert):
        """Dispatch alert to registered handlers."""
        
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {str(e)}")
    
    def add_alert_handler(self, handler: Callable[[CostAlert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    async def _check_immediate_thresholds(self, metrics: RequestMetrics):
        """Check for immediate threshold violations."""
        
        # Check for unusually expensive single request
        if metrics.cost_usd > self.config.get("single_request_cost_threshold", 0.5):
            await self._create_alert(
                AlertLevel.WARNING,
                f"Expensive request: ${metrics.cost_usd:.4f} for {metrics.model_name}",
                CostLimitType.TOTAL,
                metrics.cost_usd,
                self.config.get("single_request_cost_threshold", 0.5),
                ["Review model selection", "Check prompt efficiency"]
            )
        
        # Check for high token usage
        if metrics.total_tokens > self.config.get("high_token_threshold", 8000):
            await self._create_alert(
                AlertLevel.INFO,
                f"High token usage: {metrics.total_tokens} tokens for {metrics.model_name}",
                CostLimitType.TOTAL,
                metrics.cost_usd,
                0.0,
                ["Optimize prompt length", "Consider prompt summarization"]
            )
    
    async def _analyze_cost_trends(self):
        """Analyze cost trends for pattern detection."""
        
        try:
            # Get 7 days of hourly usage data
            usage_data = await self._get_usage_data(24 * 7)
            
            if len(usage_data) < 24:  # Need at least 24 hours of data
                return
            
            # Group by hour and calculate hourly costs
            hourly_costs = defaultdict(float)
            for metrics in usage_data:
                hour_key = metrics.timestamp.strftime("%Y%m%d_%H")
                hourly_costs[hour_key] += metrics.cost_usd
            
            # Convert to time series
            sorted_hours = sorted(hourly_costs.keys())
            cost_series = [hourly_costs[hour] for hour in sorted_hours]
            
            # Detect trends
            if len(cost_series) >= 24:
                # Calculate trend using linear regression
                x = np.arange(len(cost_series))
                slope, intercept = np.polyfit(x, cost_series, 1)
                
                trend_direction = "stable"
                if slope > 0.001:  # Increasing by more than $0.001 per hour
                    trend_direction = "increasing"
                elif slope < -0.001:  # Decreasing by more than $0.001 per hour
                    trend_direction = "decreasing"
                
                # Store trend analysis
                self.cost_patterns["hourly_trend"] = {
                    "direction": trend_direction,
                    "slope": slope,
                    "average_hourly_cost": statistics.mean(cost_series),
                    "cost_variance": statistics.variance(cost_series) if len(cost_series) > 1 else 0
                }
                
                # Alert on rapid cost increases
                if slope > 0.01:  # Rapid increase
                    await self._create_alert(
                        AlertLevel.WARNING,
                        f"Rapid cost increase detected: +${slope*24:.4f}/day trend",
                        CostLimitType.DAILY,
                        slope * 24,
                        0.0,
                        ["Investigate cause of cost increase", "Review recent changes"]
                    )
                
        except Exception as e:
            logger.error(f"Cost trend analysis failed: {str(e)}")
    
    async def _generate_optimization_recommendations(self):
        """Generate cost optimization recommendations."""
        
        try:
            # Get recent usage data
            usage_data = await self._get_usage_data(24 * 3)  # 3 days
            
            if len(usage_data) < 10:
                return
            
            recommendations = []
            
            # Model usage analysis
            model_costs = defaultdict(float)
            model_requests = defaultdict(int)
            
            for metrics in usage_data:
                model_costs[metrics.model_name] += metrics.cost_usd
                model_requests[metrics.model_name] += 1
            
            total_cost = sum(model_costs.values())
            
            # Identify expensive models with low utilization
            for model_name, cost in model_costs.items():
                if cost > total_cost * 0.4 and model_requests[model_name] < 10:
                    recommendations.append({
                        "type": "model_optimization",
                        "priority": "high",
                        "description": f"Model {model_name} has high cost but low usage",
                        "potential_savings_usd": cost * 0.6,
                        "action": "Consider switching to cheaper alternative model"
                    })
            
            # Token efficiency analysis
            high_token_requests = [m for m in usage_data if m.total_tokens > 2000]
            if len(high_token_requests) > len(usage_data) * 0.3:
                avg_token_cost = statistics.mean([m.cost_usd for m in high_token_requests])
                recommendations.append({
                    "type": "token_optimization",
                    "priority": "medium",
                    "description": "High proportion of requests use many tokens",
                    "potential_savings_usd": avg_token_cost * len(high_token_requests) * 0.3,
                    "action": "Implement prompt optimization and summarization"
                })
            
            # Cache opportunity analysis
            similar_prompts = await self._identify_similar_prompts(usage_data)
            if similar_prompts:
                cache_savings = sum(metrics.cost_usd for metrics in similar_prompts) * 0.8
                recommendations.append({
                    "type": "caching_opportunity",
                    "priority": "medium",
                    "description": f"Found {len(similar_prompts)} similar requests",
                    "potential_savings_usd": cache_savings,
                    "action": "Improve caching strategy for repeated requests"
                })
            
            # Store recommendations
            self.optimization_rules = recommendations
            
        except Exception as e:
            logger.error(f"Optimization recommendation generation failed: {str(e)}")
    
    async def _identify_similar_prompts(self, usage_data: List[RequestMetrics]) -> List[RequestMetrics]:
        """Identify requests with similar prompts that could benefit from caching."""
        
        # Simple similarity detection based on prompt length and keywords
        # In practice, this would use more sophisticated NLP techniques
        
        similar_groups = defaultdict(list)
        
        for metrics in usage_data:
            # Create simple signature based on prompt characteristics
            # This is a placeholder - would use actual prompt data
            signature = f"tokens_{metrics.total_tokens//100}00_model_{metrics.model_name}"
            similar_groups[signature].append(metrics)
        
        # Return groups with multiple similar requests
        similar_requests = []
        for group in similar_groups.values():
            if len(group) >= 3:  # 3 or more similar requests
                similar_requests.extend(group)
        
        return similar_requests
    
    async def _update_forecasts(self):
        """Update cost forecasts for all budget periods."""
        
        for period in self.budget_limits.keys():
            await self._generate_forecast(period)
    
    async def _generate_forecast(self, period: CostLimitType) -> BudgetForecast:
        """Generate cost forecast for specified period."""
        
        try:
            current_usage = await self._get_current_usage(period)
            budget_limit = self.budget_limits[period]
            
            # Get historical data for trend analysis
            if period == CostLimitType.HOURLY:
                historical_hours = 24
                period_hours = 1
            elif period == CostLimitType.DAILY:
                historical_hours = 24 * 7
                period_hours = 24
            elif period == CostLimitType.WEEKLY:
                historical_hours = 24 * 7 * 4
                period_hours = 24 * 7
            elif period == CostLimitType.MONTHLY:
                historical_hours = 24 * 7 * 12
                period_hours = 24 * 7 * 4
            else:  # TOTAL
                historical_hours = 24 * 7 * 52
                period_hours = 24 * 365
            
            usage_data = await self._get_usage_data(historical_hours)
            
            if not usage_data:
                # No data, create conservative forecast
                forecast = BudgetForecast(
                    period_type=period,
                    current_usage=current_usage,
                    budget_limit=budget_limit,
                    projected_usage=current_usage,
                    projected_end_date=datetime.now() + timedelta(hours=period_hours),
                    confidence_score=0.1,
                    trend_direction="stable"
                )
                self.cost_forecasts[period] = forecast
                return forecast
            
            # Calculate spending rate
            spending_rates = []
            hourly_costs = defaultdict(float)
            
            for metrics in usage_data:
                hour_key = metrics.timestamp.strftime("%Y%m%d_%H")
                hourly_costs[hour_key] += metrics.cost_usd
            
            recent_hourly_costs = list(hourly_costs.values())[-24:]  # Last 24 hours
            avg_hourly_rate = statistics.mean(recent_hourly_costs) if recent_hourly_costs else 0
            
            # Project usage
            remaining_hours = period_hours - (
                (datetime.now() - self._get_period_start(period, datetime.now())).total_seconds() / 3600
            )
            
            projected_additional_cost = avg_hourly_rate * remaining_hours
            projected_total = current_usage + projected_additional_cost
            
            # Determine trend direction
            if len(recent_hourly_costs) >= 12:
                # Compare first half vs second half
                first_half = recent_hourly_costs[:len(recent_hourly_costs)//2]
                second_half = recent_hourly_costs[len(recent_hourly_costs)//2:]
                
                avg_first = statistics.mean(first_half)
                avg_second = statistics.mean(second_half)
                
                if avg_second > avg_first * 1.2:
                    trend_direction = "increasing"
                elif avg_second < avg_first * 0.8:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
            
            # Calculate confidence based on data amount and variance
            confidence_score = min(1.0, len(usage_data) / 100)  # More data = higher confidence
            if recent_hourly_costs:
                cost_variance = statistics.variance(recent_hourly_costs)
                # Lower variance = higher confidence
                variance_factor = max(0.1, 1.0 - (cost_variance / (avg_hourly_rate + 0.001)))
                confidence_score *= variance_factor
            
            # Calculate days until budget exhaustion
            days_until_exhaustion = None
            if avg_hourly_rate > 0 and projected_total > budget_limit:
                remaining_budget = budget_limit - current_usage
                hours_until_exhaustion = remaining_budget / avg_hourly_rate
                days_until_exhaustion = int(hours_until_exhaustion / 24) if hours_until_exhaustion > 0 else 0
            
            forecast = BudgetForecast(
                period_type=period,
                current_usage=current_usage,
                budget_limit=budget_limit,
                projected_usage=projected_total,
                projected_end_date=datetime.now() + timedelta(hours=remaining_hours),
                confidence_score=confidence_score,
                trend_direction=trend_direction,
                days_until_exhaustion=days_until_exhaustion
            )
            
            self.cost_forecasts[period] = forecast
            
            # Generate alert if budget will be exceeded
            if projected_total > budget_limit and confidence_score > 0.5:
                overage = projected_total - budget_limit
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"{period.value.title()} budget forecast exceeded by ${overage:.4f}",
                    period,
                    current_usage,
                    budget_limit,
                    ["Reduce usage", "Optimize model selection", "Increase budget if necessary"]
                )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Forecast generation failed for {period.value}: {str(e)}")
            
            # Return error forecast
            return BudgetForecast(
                period_type=period,
                current_usage=current_usage,
                budget_limit=budget_limit,
                projected_usage=current_usage,
                projected_end_date=datetime.now(),
                confidence_score=0.0,
                trend_direction="unknown"
            )
    
    def _get_period_start(self, period: CostLimitType, reference_time: datetime) -> datetime:
        """Get start time for budget period."""
        
        if period == CostLimitType.HOURLY:
            return reference_time.replace(minute=0, second=0, microsecond=0)
        elif period == CostLimitType.DAILY:
            return reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == CostLimitType.WEEKLY:
            days_back = reference_time.weekday()
            return reference_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
        elif period == CostLimitType.MONTHLY:
            return reference_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # TOTAL
            return datetime.min
    
    async def _update_all_forecasts(self):
        """Update all budget forecasts."""
        
        for period in self.budget_limits.keys():
            try:
                await self._generate_forecast(period)
            except Exception as e:
                logger.error(f"Failed to update forecast for {period.value}: {str(e)}")
    
    async def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status across all periods."""
        
        status_report = {
            "overall_status": BudgetStatus.HEALTHY.value,
            "budget_periods": {},
            "alerts_summary": {
                "total_active_alerts": 0,
                "critical_alerts": 0,
                "recent_alerts": []
            },
            "forecasts": {},
            "optimization_opportunities": self.optimization_rules
        }
        
        worst_status = BudgetStatus.HEALTHY
        
        try:
            # Check each budget period
            for period, budget_limit in self.budget_limits.items():
                current_usage = await self._get_current_usage(period)
                utilization_percent = (current_usage / budget_limit) * 100
                
                # Determine status
                if utilization_percent >= 100:
                    period_status = BudgetStatus.EXCEEDED
                elif utilization_percent >= 90:
                    period_status = BudgetStatus.CRITICAL
                elif utilization_percent >= 75:
                    period_status = BudgetStatus.HIGH
                elif utilization_percent >= 50:
                    period_status = BudgetStatus.MODERATE
                else:
                    period_status = BudgetStatus.HEALTHY
                
                # Update worst status
                status_values = {
                    BudgetStatus.HEALTHY: 0,
                    BudgetStatus.MODERATE: 1,
                    BudgetStatus.HIGH: 2,
                    BudgetStatus.CRITICAL: 3,
                    BudgetStatus.EXCEEDED: 4
                }
                
                if status_values[period_status] > status_values[worst_status]:
                    worst_status = period_status
                
                status_report["budget_periods"][period.value] = {
                    "budget_limit": budget_limit,
                    "current_usage": current_usage,
                    "utilization_percent": utilization_percent,
                    "remaining_budget": budget_limit - current_usage,
                    "status": period_status.value
                }
                
                # Add forecast if available
                if period in self.cost_forecasts:
                    forecast = self.cost_forecasts[period]
                    status_report["forecasts"][period.value] = {
                        "projected_usage": forecast.projected_usage,
                        "projected_end_date": forecast.projected_end_date.isoformat(),
                        "confidence_score": forecast.confidence_score,
                        "trend_direction": forecast.trend_direction,
                        "days_until_exhaustion": forecast.days_until_exhaustion,
                        "will_exceed_budget": forecast.projected_usage > forecast.budget_limit
                    }
            
            status_report["overall_status"] = worst_status.value
            
            # Alert summary
            recent_alerts = [a for a in self.alert_history if a.timestamp >= datetime.now() - timedelta(hours=24)]
            critical_alerts = [a for a in recent_alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
            
            status_report["alerts_summary"] = {
                "total_active_alerts": len(recent_alerts),
                "critical_alerts": len(critical_alerts),
                "recent_alerts": [
                    {
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts[-5:]  # Last 5 alerts
                ]
            }
            
            return status_report
            
        except Exception as e:
            logger.error(f"Failed to get budget status: {str(e)}")
            return {"error": str(e)}
    
    async def get_cost_insights(self, analysis_depth: str = "standard") -> Dict[str, Any]:
        """Generate detailed cost insights and recommendations."""
        
        try:
            insights = {
                "analysis_depth": analysis_depth,
                "timestamp": datetime.now().isoformat(),
                "cost_efficiency": {},
                "usage_patterns": {},
                "optimization_recommendations": [],
                "risk_assessment": {}
            }
            
            # Get usage data based on analysis depth
            if analysis_depth == "quick":
                usage_data = await self._get_usage_data(6)  # 6 hours
            elif analysis_depth == "deep":
                usage_data = await self._get_usage_data(24 * 30)  # 30 days
            else:  # standard
                usage_data = await self._get_usage_data(24 * 7)  # 7 days
            
            if not usage_data:
                return {"message": "Insufficient data for cost insights"}
            
            # Cost efficiency analysis
            total_cost = sum(m.cost_usd for m in usage_data)
            successful_requests = [m for m in usage_data if m.success]
            
            if successful_requests:
                avg_cost_per_success = total_cost / len(successful_requests)
                efficiency_score = min(100, (0.01 / avg_cost_per_success) * 100) if avg_cost_per_success > 0 else 100
                
                insights["cost_efficiency"] = {
                    "average_cost_per_request": avg_cost_per_success,
                    "efficiency_score": efficiency_score,
                    "total_successful_requests": len(successful_requests),
                    "success_rate": len(successful_requests) / len(usage_data)
                }
            
            # Usage pattern analysis
            hourly_distribution = defaultdict(float)
            model_distribution = defaultdict(float)
            
            for metrics in usage_data:
                hour = metrics.timestamp.hour
                hourly_distribution[hour] += metrics.cost_usd
                model_distribution[metrics.model_name] += metrics.cost_usd
            
            # Find peak usage hours
            peak_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            
            insights["usage_patterns"] = {
                "peak_usage_hours": [{"hour": hour, "cost": cost} for hour, cost in peak_hours],
                "model_cost_distribution": dict(model_distribution),
                "hourly_cost_variance": statistics.variance(list(hourly_distribution.values())) if len(hourly_distribution) > 1 else 0
            }
            
            # Risk assessment
            insights["risk_assessment"] = await self._assess_cost_risks(usage_data)
            
            # Consolidate optimization recommendations
            insights["optimization_recommendations"] = self.optimization_rules
            
            return insights
            
        except Exception as e:
            logger.error(f"Cost insights generation failed: {str(e)}")
            return {"error": str(e)}
    
    async def _assess_cost_risks(self, usage_data: List[RequestMetrics]) -> Dict[str, Any]:
        """Assess cost-related risks."""
        
        risks = {
            "risk_level": "low",
            "risk_factors": [],
            "mitigation_suggestions": []
        }
        
        try:
            # High variance risk
            costs = [m.cost_usd for m in usage_data if m.cost_usd > 0]
            if costs:
                cost_variance = statistics.variance(costs)
                avg_cost = statistics.mean(costs)
                
                if cost_variance > avg_cost * 2:  # High variance
                    risks["risk_factors"].append("High cost variance - unpredictable spending")
                    risks["mitigation_suggestions"].append("Implement stricter cost controls")
            
            # Model concentration risk
            model_costs = defaultdict(float)
            for metrics in usage_data:
                model_costs[metrics.model_name] += metrics.cost_usd
            
            total_cost = sum(model_costs.values())
            if total_cost > 0:
                max_model_share = max(model_costs.values()) / total_cost
                
                if max_model_share > 0.8:  # Single model dominates
                    risks["risk_factors"].append("High dependency on single expensive model")
                    risks["mitigation_suggestions"].append("Diversify model usage")
            
            # Error rate risk
            error_rate = len([m for m in usage_data if not m.success]) / len(usage_data)
            if error_rate > 0.1:  # > 10% error rate
                risks["risk_factors"].append("High error rate leading to wasted costs")
                risks["mitigation_suggestions"].append("Improve error handling and retry logic")
            
            # Determine overall risk level
            if len(risks["risk_factors"]) >= 3:
                risks["risk_level"] = "high"
            elif len(risks["risk_factors"]) >= 2:
                risks["risk_level"] = "medium"
            elif len(risks["risk_factors"]) >= 1:
                risks["risk_level"] = "low"
            else:
                risks["risk_level"] = "minimal"
            
        except Exception as e:
            logger.warning(f"Risk assessment failed: {str(e)}")
            risks["error"] = str(e)
        
        return risks
    
    async def export_cost_report(
        self,
        start_date: datetime,
        end_date: datetime,
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """Export detailed cost report for specified date range."""
        
        try:
            # Get usage data for date range
            usage_data = await self._get_usage_data_in_range(start_date, end_date)
            
            if not usage_data:
                return {"message": "No usage data found for specified date range"}
            
            # Generate comprehensive report
            report = {
                "report_metadata": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "generated_at": datetime.now().isoformat(),
                    "total_records": len(usage_data)
                },
                "cost_summary": self._generate_cost_summary(usage_data),
                "provider_analysis": self._analyze_provider_costs(usage_data),
                "model_analysis": self._analyze_model_costs(usage_data),
                "time_series_analysis": self._generate_time_series_analysis(usage_data),
                "efficiency_metrics": self._calculate_efficiency_metrics(usage_data),
                "recommendations": await self._generate_report_recommendations(usage_data)
            }
            
            if format_type == "csv":
                # Convert to CSV format (simplified)
                csv_data = []
                for metrics in usage_data:
                    csv_data.append({
                        "timestamp": metrics.timestamp.isoformat(),
                        "model": metrics.model_name,
                        "provider": metrics.provider.value,
                        "tokens": metrics.total_tokens,
                        "cost_usd": metrics.cost_usd,
                        "success": metrics.success
                    })
                
                report["csv_data"] = csv_data
            
            return report
            
        except Exception as e:
            logger.error(f"Cost report export failed: {str(e)}")
            return {"error": str(e)}
    
    async def _get_usage_data_in_range(self, start_date: datetime, end_date: datetime) -> List[RequestMetrics]:
        """Get usage data within specified date range."""
        
        # This would query the database for actual metrics within date range
        # For now, return empty list as placeholder
        return []
    
    def _generate_cost_summary(self, usage_data: List[RequestMetrics]) -> Dict[str, Any]:
        """Generate cost summary from usage data."""
        
        total_cost = sum(m.cost_usd for m in usage_data)
        total_tokens = sum(m.total_tokens for m in usage_data)
        successful_requests = [m for m in usage_data if m.success]
        
        return {
            "total_cost_usd": total_cost,
            "total_requests": len(usage_data),
            "successful_requests": len(successful_requests),
            "total_tokens": total_tokens,
            "average_cost_per_request": total_cost / len(usage_data) if usage_data else 0,
            "average_cost_per_successful_request": total_cost / len(successful_requests) if successful_requests else 0,
            "cost_per_1k_tokens": (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
        }
    
    def _analyze_provider_costs(self, usage_data: List[RequestMetrics]) -> Dict[str, Any]:
        """Analyze costs by provider."""
        
        provider_stats = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_requests": 0,
            "total_tokens": 0,
            "success_rate": 0.0,
            "avg_latency_ms": 0.0
        })
        
        for metrics in usage_data:
            stats = provider_stats[metrics.provider.value]
            stats["total_cost"] += metrics.cost_usd
            stats["total_requests"] += 1
            stats["total_tokens"] += metrics.total_tokens
            
            if metrics.success:
                stats["avg_latency_ms"] = (stats["avg_latency_ms"] + metrics.latency_ms) / 2
        
        # Calculate success rates
        for provider_name, stats in provider_stats.items():
            provider_successes = len([m for m in usage_data if m.provider.value == provider_name and m.success])
            stats["success_rate"] = provider_successes / stats["total_requests"] if stats["total_requests"] > 0 else 0
        
        return dict(provider_stats)
    
    def _analyze_model_costs(self, usage_data: List[RequestMetrics]) -> Dict[str, Any]:
        """Analyze costs by model."""
        
        model_stats = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_requests": 0,
            "average_cost": 0.0,
            "cost_efficiency_rank": 0
        })
        
        for metrics in usage_data:
            stats = model_stats[metrics.model_name]
            stats["total_cost"] += metrics.cost_usd
            stats["total_requests"] += 1
        
        # Calculate averages and rankings
        for model_name, stats in model_stats.items():
            stats["average_cost"] = stats["total_cost"] / stats["total_requests"]
        
        # Rank by cost efficiency (lower cost per request = higher rank)
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["average_cost"])
        for rank, (model_name, stats) in enumerate(sorted_models):
            stats["cost_efficiency_rank"] = rank + 1
        
        return dict(model_stats)
    
    def _generate_time_series_analysis(self, usage_data: List[RequestMetrics]) -> Dict[str, Any]:
        """Generate time series analysis of costs."""
        
        # Group by hour
        hourly_costs = defaultdict(float)
        hourly_requests = defaultdict(int)
        
        for metrics in usage_data:
            hour_key = metrics.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_costs[hour_key] += metrics.cost_usd
            hourly_requests[hour_key] += 1
        
        # Sort chronologically
        sorted_hours = sorted(hourly_costs.keys())
        cost_series = [hourly_costs[hour] for hour in sorted_hours]
        request_series = [hourly_requests[hour] for hour in sorted_hours]
        
        time_series = {
            "hourly_costs": [{"hour": hour, "cost": hourly_costs[hour]} for hour in sorted_hours],
            "cost_trend": "stable",
            "peak_hour": None,
            "lowest_hour": None
        }
        
        if cost_series:
            # Find peak and lowest hours
            max_cost_idx = cost_series.index(max(cost_series))
            min_cost_idx = cost_series.index(min(cost_series))
            
            time_series["peak_hour"] = {
                "hour": sorted_hours[max_cost_idx],
                "cost": cost_series[max_cost_idx]
            }
            
            time_series["lowest_hour"] = {
                "hour": sorted_hours[min_cost_idx],
                "cost": cost_series[min_cost_idx]
            }
            
            # Determine trend
            if len(cost_series) >= 6:
                first_half = cost_series[:len(cost_series)//2]
                second_half = cost_series[len(cost_series)//2:]
                
                avg_first = statistics.mean(first_half)
                avg_second = statistics.mean(second_half)
                
                if avg_second > avg_first * 1.2:
                    time_series["cost_trend"] = "increasing"
                elif avg_second < avg_first * 0.8:
                    time_series["cost_trend"] = "decreasing"
        
        return time_series
    
    def _calculate_efficiency_metrics(self, usage_data: List[RequestMetrics]) -> Dict[str, Any]:
        """Calculate cost efficiency metrics."""
        
        if not usage_data:
            return {}
        
        successful_requests = [m for m in usage_data if m.success]
        failed_requests = [m for m in usage_data if not m.success]
        
        # Basic efficiency metrics
        total_cost = sum(m.cost_usd for m in usage_data)
        wasted_cost = sum(m.cost_usd for m in failed_requests)
        
        efficiency_metrics = {
            "cost_effectiveness": {
                "total_cost": total_cost,
                "wasted_cost": wasted_cost,
                "waste_percentage": (wasted_cost / total_cost * 100) if total_cost > 0 else 0,
                "effective_cost": total_cost - wasted_cost
            },
            "token_efficiency": {
                "average_tokens_per_request": statistics.mean([m.total_tokens for m in successful_requests]) if successful_requests else 0,
                "cost_per_1k_tokens": (total_cost / sum(m.total_tokens for m in usage_data) * 1000) if sum(m.total_tokens for m in usage_data) > 0 else 0
            },
            "performance_efficiency": {
                "average_latency_ms": statistics.mean([m.latency_ms for m in successful_requests]) if successful_requests else 0,
                "requests_per_dollar": len(successful_requests) / total_cost if total_cost > 0 else 0
            }
        }
        
        return efficiency_metrics
    
    async def _generate_report_recommendations(self, usage_data: List[RequestMetrics]) -> List[Dict[str, Any]]:
        """Generate recommendations for cost report."""
        
        recommendations = []
        
        if not usage_data:
            return recommendations
        
        try:
            # Model optimization recommendations
            model_costs = defaultdict(float)
            model_requests = defaultdict(int)
            
            for metrics in usage_data:
                model_costs[metrics.model_name] += metrics.cost_usd
                model_requests[metrics.model_name] += 1
            
            # Find expensive models with high usage
            total_cost = sum(model_costs.values())
            
            for model_name, cost in model_costs.items():
                if cost > total_cost * 0.4:  # Model accounts for > 40% of costs
                    recommendations.append({
                        "priority": "high",
                        "category": "model_optimization",
                        "title": f"Optimize usage of expensive model: {model_name}",
                        "description": f"Model accounts for {(cost/total_cost)*100:.1f}% of total costs",
                        "potential_impact": "20-40% cost reduction",
                        "actions": [
                            "Review if cheaper models can handle same tasks",
                            "Implement task complexity routing",
                            "Consider prompt optimization"
                        ]
                    })
            
            # Error rate optimization
            error_rate = len([m for m in usage_data if not m.success]) / len(usage_data)
            if error_rate > 0.05:  # > 5% error rate
                wasted_cost = sum(m.cost_usd for m in usage_data if not m.success)
                recommendations.append({
                    "priority": "medium",
                    "category": "reliability",
                    "title": "Reduce request failures",
                    "description": f"Error rate: {error_rate*100:.1f}%, wasted cost: ${wasted_cost:.4f}",
                    "potential_impact": f"${wasted_cost:.4f} cost savings",
                    "actions": [
                        "Improve error handling",
                        "Add request validation",
                        "Implement better retry logic"
                    ]
                })
            
            # Token efficiency recommendations
            avg_tokens = statistics.mean([m.total_tokens for m in usage_data])
            if avg_tokens > 2000:  # High token usage
                recommendations.append({
                    "priority": "medium",
                    "category": "efficiency",
                    "title": "Optimize token usage",
                    "description": f"High average token usage: {avg_tokens:.0f} tokens/request",
                    "potential_impact": "15-30% cost reduction",
                    "actions": [
                        "Implement prompt compression",
                        "Use shorter, more focused prompts",
                        "Consider prompt templates"
                    ]
                })
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {str(e)}")
        
        return recommendations
