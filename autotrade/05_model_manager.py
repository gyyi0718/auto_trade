#!/usr/bin/env python3
"""
🚀 모델 자동 관리 시스템
- 주기적 자동 재학습
- A/B 테스트로 안전한 모델 교체
- 버전 관리 및 롤백
- 프로덕션 배포 자동화
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import sys

# 같은 디렉토리의 model_evaluator import
try:
    from model_evaluator import ModelEvaluator
except ImportError:
    print("⚠️ model_evaluator.py를 같은 디렉토리에 두세요")
    sys.exit(1)


class ModelManager:
    """모델 버전 관리 및 자동 업데이트"""
    
    def __init__(self, base_dir: str = "./production"):
        self.base_dir = Path(base_dir)
        
        # 디렉토리 구조
        self.active_dir = self.base_dir / "active"          # 현재 운영 중인 모델
        self.candidates_dir = self.base_dir / "candidates"  # 후보 모델들
        self.history_dir = self.base_dir / "history"        # 과거 버전들
        self.ab_test_dir = self.base_dir / "ab_test"        # A/B 테스트 중
        
        # 디렉토리 생성
        for d in [self.active_dir, self.candidates_dir, self.history_dir, self.ab_test_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 설정 파일
        self.config_path = self.base_dir / "config.json"
        self.history_path = self.base_dir / "deployment_history.json"
        
        self.load_config()
        self.load_history()
    
    def load_config(self):
        """설정 로드"""
        default_config = {
            'auto_update_enabled': True,
            'min_score_improvement': 5.0,  # 최소 5점 개선되어야 교체
            'ab_test_days': 7,             # A/B 테스트 7일
            'training_days': 30,           # 최근 30일 데이터로 학습
            'top_coins': 5,                # 상위 5개 코인
            'force_update': False          # 강제 업데이트 (검증 없이)
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """설정 저장"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_history(self):
        """배포 히스토리 로드"""
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
    
    def save_history(self):
        """배포 히스토리 저장"""
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_active_model(self) -> Path:
        """현재 운영 중인 모델 경로"""
        models = list(self.active_dir.glob("*.pkl")) + list(self.active_dir.glob("*.joblib"))
        
        if models:
            return sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        return None
    
    def train_new_model(self, days: int = 30, top: int = 5) -> Path:
        """
        새 모델 학습 (pipeline.py 실행)
        """
        print(f"\n{'='*60}")
        print(f"🔄 새 모델 학습 시작")
        print(f"   데이터: 최근 {days}일")
        print(f"   코인: 상위 {top}개")
        print(f"{'='*60}\n")
        
        # pipeline.py 실행
        try:
            cmd = [
                sys.executable, "pipeline.py",
                "--top", str(top),
                "--days", str(days)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                print(f"❌ 학습 실패:\n{result.stderr}")
                return None
            
            print("✅ 학습 완료")
            
            # 생성된 모델 찾기 (./models 디렉토리에서 가장 최근 것)
            models_dir = Path("./models")
            if not models_dir.exists():
                print("❌ ./models 디렉토리가 없습니다")
                return None
            
            models = sorted(
                models_dir.glob("*.pkl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not models:
                print("❌ 생성된 모델을 찾을 수 없습니다")
                return None
            
            new_model = models[0]
            print(f"✅ 새 모델: {new_model.name}")
            
            # 후보 디렉토리로 복사
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate_path = self.candidates_dir / f"model_{timestamp}.pkl"
            
            shutil.copy2(new_model, candidate_path)
            
            # 메타데이터도 복사
            meta_src = new_model.with_suffix('.json')
            if meta_src.exists():
                meta_dst = candidate_path.with_suffix('.json')
                shutil.copy2(meta_src, meta_dst)
            
            print(f"✅ 후보 저장: {candidate_path.name}\n")
            
            return candidate_path
            
        except subprocess.TimeoutExpired:
            print("❌ 학습 시간 초과 (30분)")
            return None
        except Exception as e:
            print(f"❌ 학습 오류: {e}")
            return None
    
    def compare_models(self, model_a: Path, model_b: Path) -> dict:
        """두 모델 비교"""
        evaluator = ModelEvaluator()
        
        result = evaluator.compare_two_models(model_a, model_b)
        
        return result
    
    def deploy_model(self, model_path: Path, reason: str = "Manual deployment"):
        """
        모델을 프로덕션으로 배포
        """
        print(f"\n{'='*60}")
        print(f"🚀 모델 배포")
        print(f"{'='*60}\n")
        
        if not model_path.exists():
            print(f"❌ 모델 없음: {model_path}")
            return False
        
        # 현재 운영 중인 모델 백업
        current_model = self.get_active_model()
        
        if current_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.history_dir / f"backup_{timestamp}_{current_model.name}"
            
            print(f"💾 현재 모델 백업: {current_model.name}")
            shutil.copy2(current_model, backup_path)
            
            # 메타데이터도 백업
            meta_src = current_model.with_suffix('.json')
            if meta_src.exists():
                meta_dst = backup_path.with_suffix('.json')
                shutil.copy2(meta_src, meta_dst)
            
            # 기존 모델 삭제
            current_model.unlink()
            if meta_src.exists():
                meta_src.unlink()
        
        # 새 모델 배포
        deployed_path = self.active_dir / model_path.name
        shutil.copy2(model_path, deployed_path)
        
        # 메타데이터도 복사
        meta_src = model_path.with_suffix('.json')
        if meta_src.exists():
            meta_dst = deployed_path.with_suffix('.json')
            shutil.copy2(meta_src, meta_dst)
        
        print(f"✅ 새 모델 배포: {model_path.name}")
        
        # 히스토리 기록
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'model': str(model_path),
            'deployed_to': str(deployed_path),
            'reason': reason,
            'previous_model': str(current_model) if current_model else None
        })
        
        self.save_history()
        
        print(f"{'='*60}\n")
        
        return True
    
    def auto_update(self, force: bool = False):
        """
        자동 업데이트 프로세스
        1. 새 모델 학습
        2. 현재 모델과 비교 (force=False인 경우)
        3. 개선되면 배포
        """
        print(f"\n{'='*60}")
        print(f"🤖 자동 업데이트 시작")
        print(f"   모드: {'강제' if force else 'A/B 테스트'}")
        print(f"{'='*60}\n")
        
        # 1. 새 모델 학습
        new_model = self.train_new_model(
            days=self.config['training_days'],
            top=self.config['top_coins']
        )
        
        if not new_model:
            print("❌ 새 모델 학습 실패")
            return False
        
        # 2. 강제 모드면 즉시 배포
        if force:
            print("⚡ 강제 모드 - 검증 없이 배포\n")
            return self.deploy_model(new_model, reason="Forced auto-update")
        
        # 3. 현재 모델과 비교
        current_model = self.get_active_model()
        
        if not current_model:
            print("💡 운영 중인 모델 없음 - 새 모델 배포\n")
            return self.deploy_model(new_model, reason="Initial deployment")
        
        # 4. A/B 테스트
        print(f"🧪 A/B 테스트 시작\n")
        
        comparison = self.compare_models(current_model, new_model)
        
        if not comparison:
            print("❌ 비교 실패")
            return False
        
        # 5. 판단
        min_improvement = self.config['min_score_improvement']
        score_diff = comparison['difference']
        
        if score_diff >= min_improvement:
            print(f"✅ 새 모델이 {score_diff:.2f}점 우수 (기준: {min_improvement}점)")
            print(f"   → 자동 교체 진행\n")
            return self.deploy_model(new_model, reason=f"Auto-update: +{score_diff:.2f} points")
        else:
            print(f"⚠️ 개선 부족: {score_diff:.2f}점 (기준: {min_improvement}점)")
            print(f"   → 현재 모델 유지\n")
            return False
    
    def show_history(self, n: int = 10):
        """배포 히스토리 출력"""
        print(f"\n{'='*80}")
        print(f"📜 배포 히스토리 (최근 {n}개)")
        print(f"{'='*80}\n")
        
        if not self.history:
            print("히스토리 없음\n")
            return
        
        for i, entry in enumerate(reversed(self.history[-n:]), 1):
            timestamp = entry['timestamp']
            model = Path(entry['model']).name
            reason = entry['reason']
            
            print(f"{i}. [{timestamp}]")
            print(f"   모델: {model}")
            print(f"   사유: {reason}\n")
        
        print(f"{'='*80}\n")
    
    def rollback(self):
        """이전 모델로 롤백"""
        print(f"\n{'='*60}")
        print(f"⏮️ 롤백 시작")
        print(f"{'='*60}\n")
        
        # 가장 최근 백업 찾기
        backups = sorted(self.history_dir.glob("backup_*.pkl"), reverse=True)
        
        if not backups:
            print("❌ 백업 없음\n")
            return False
        
        backup = backups[0]
        
        print(f"💾 백업 모델: {backup.name}")
        
        # 현재 모델 제거
        current = self.get_active_model()
        if current:
            current.unlink()
            meta = current.with_suffix('.json')
            if meta.exists():
                meta.unlink()
        
        # 백업 복원
        restored = self.active_dir / backup.name.replace("backup_", "").split("_", 2)[-1]
        shutil.copy2(backup, restored)
        
        # 메타데이터도 복원
        meta_backup = backup.with_suffix('.json')
        if meta_backup.exists():
            meta_restored = restored.with_suffix('.json')
            shutil.copy2(meta_backup, meta_restored)
        
        print(f"✅ 복원 완료: {restored.name}\n")
        
        # 히스토리 기록
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'model': str(backup),
            'deployed_to': str(restored),
            'reason': 'Rollback',
            'previous_model': str(current) if current else None
        })
        
        self.save_history()
        
        print(f"{'='*60}\n")
        
        return True
    
    def clean_old_backups(self, keep_days: int = 30):
        """오래된 백업 정리"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        removed = 0
        for backup in self.history_dir.glob("backup_*.pkl"):
            if datetime.fromtimestamp(backup.stat().st_mtime) < cutoff_date:
                backup.unlink()
                
                # 메타데이터도 삭제
                meta = backup.with_suffix('.json')
                if meta.exists():
                    meta.unlink()
                
                removed += 1
        
        if removed > 0:
            print(f"🗑️ {removed}개 오래된 백업 삭제 (>{keep_days}일)")


def main():
    parser = argparse.ArgumentParser(description='모델 자동 관리 시스템')
    parser.add_argument('--auto-update', action='store_true',
                       help='자동 업데이트 실행')
    parser.add_argument('--force', action='store_true',
                       help='강제 업데이트 (검증 없이)')
    parser.add_argument('--train-only', action='store_true',
                       help='학습만 하고 배포 안 함')
    parser.add_argument('--deploy', type=str,
                       help='특정 모델 배포 (경로)')
    parser.add_argument('--rollback', action='store_true',
                       help='이전 모델로 롤백')
    parser.add_argument('--show-history', action='store_true',
                       help='배포 히스토리 출력')
    parser.add_argument('--days', type=int, default=30,
                       help='학습 데이터 기간 (일)')
    parser.add_argument('--top', type=int, default=5,
                       help='상위 코인 개수')
    parser.add_argument('--clean', action='store_true',
                       help='오래된 백업 정리')
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    # 설정 업데이트
    if args.days:
        manager.config['training_days'] = args.days
    if args.top:
        manager.config['top_coins'] = args.top
    if args.force:
        manager.config['force_update'] = True
    
    manager.save_config()
    
    # 명령 실행
    if args.show_history:
        manager.show_history()
    
    elif args.rollback:
        manager.rollback()
    
    elif args.clean:
        manager.clean_old_backups()
    
    elif args.train_only:
        new_model = manager.train_new_model(args.days, args.top)
        if new_model:
            print(f"✅ 학습 완료: {new_model}")
            print(f"   후보 디렉토리: {manager.candidates_dir}\n")
    
    elif args.deploy:
        model_path = Path(args.deploy)
        if model_path.exists():
            manager.deploy_model(model_path, reason="Manual deployment")
        else:
            print(f"❌ 모델 없음: {model_path}")
    
    elif args.auto_update:
        success = manager.auto_update(force=args.force)
        
        if success:
            print("✅ 자동 업데이트 성공\n")
        else:
            print("⚠️ 자동 업데이트 실패 또는 유지\n")
    
    else:
        # 기본: 현재 상태 출력
        print(f"\n{'='*60}")
        print(f"📊 현재 상태")
        print(f"{'='*60}\n")
        
        current = manager.get_active_model()
        if current:
            print(f"✅ 운영 중인 모델: {current.name}")
            print(f"   경로: {current}")
        else:
            print(f"⚠️ 운영 중인 모델 없음")
        
        print(f"\n설정:")
        print(f"  • 자동 업데이트: {'ON' if manager.config['auto_update_enabled'] else 'OFF'}")
        print(f"  • 최소 개선: {manager.config['min_score_improvement']}점")
        print(f"  • 학습 기간: {manager.config['training_days']}일")
        print(f"  • 상위 코인: {manager.config['top_coins']}개")
        
        print(f"\n디렉토리:")
        print(f"  • 운영: {manager.active_dir}")
        print(f"  • 후보: {manager.candidates_dir}")
        print(f"  • 백업: {manager.history_dir}")
        
        print(f"\n{'='*60}\n")
        
        print("사용법:")
        print("  python model_manager.py --auto-update       # 자동 업데이트")
        print("  python model_manager.py --auto-update --force   # 강제 업데이트")
        print("  python model_manager.py --train-only        # 학습만")
        print("  python model_manager.py --show-history      # 히스토리")
        print("  python model_manager.py --rollback          # 롤백")
        print()


if __name__ == "__main__":
    main()