## 功能说明

`gitee_pr_stat.py` 是一个工具，用于从 Gitee 仓库中查询 Pull Request (PR) 的详细信息，并导出统计数据。支持导出为 CSV 和 JSON 格式，同时可以分析合并的 PR 的源码文件变更统计。

## Install
```shell
git clone https://github.com/hhqx/py3_tools.git
cd py3_tools
pip install .[gitee]
```

## 快速开始
以下命令展示了如何使用工具查询 PR 数据并导出结果：
```shell
python -m py3_tools.gitees.gitee_pr_stat \
    --owner ascend --repo msit --author hhhqx \
    --state all \
    --since "2025-01-01" \
    --until "2025-12-31" \
    --per_page 100 \
    --max_pages 5 \
    --output_dir "./results/gitee"
```

## 参数说明
| 参数名         | 类型       | 默认值       | 说明                                                                 |
|----------------|------------|--------------|----------------------------------------------------------------------|
| `--owner`      | `str`      | 必填         | 仓库所属用户或组织名，例如 `ascend`。                                 |
| `--repo`       | `str`      | 必填         | 仓库名称，例如 `msit`。                                              |
| `--author`     | `str`      | 必填         | PR 作者的 Gitee 用户名，例如 `hhhqx`。                               |
| `--state`      | `str`      | `all`        | PR 状态，可选值：`open`、`closed`、`merged`、`all`。                 |
| `--since`      | `datetime` | 无           | 查询指定日期之后创建或更新的 PR，格式为 `YYYY-MM-DD`。               |
| `--until`      | `datetime` | 无           | 查询指定日期之前创建或更新的 PR，格式为 `YYYY-MM-DD`。               |
| `--per-page`   | `int`      | `50`         | 每页返回的 PR 数量，最大值为 `100`。                                 |
| `--max-pages`  | `int`      | `5`          | 查询的最大页数。                                                     |
| `--output-dir` | `str`      | `output`     | 导出文件的保存目录，例如 `./results/gitee`。                         |

## 实现细节
1. **PR 数据查询**：
   - 使用 Gitee API 查询 PR 数据。
   - 支持按作者、状态、时间范围等条件过滤。
2. **数据导出**：
   - 支持导出为 CSV 和 JSON 格式。
   - 导出的数据包括 PR 的基本信息和文件变更统计。
3. **合并 PR 分析**：
   - 使用 pandas 分析合并的 PR 的源码文件变更统计。
   - 统计每个 PR 的新增行数、删除行数、变更文件数等。
   - 导出详细统计和汇总统计到 CSV 文件。

## 示例输出
### 导出的 CSV 文件
#### PR 详情文件 (`gitee_prs_ascend_msit_hhhqx.csv`)
| PR Number | PR Title                  | PR State | Created At | Updated At | File Name                          | Change Type | Added Lines | Removed Lines | Total Changed Lines |
|-----------|---------------------------|----------|------------|------------|------------------------------------|-------------|-------------|---------------|---------------------|
| 1234      | Fix bug in module         | merged   | 2025-01-01 | 2025-01-02 | src/module.py                      | modified    | 10          | 2             | 12                  |
| 1235      | Add new feature           | merged   | 2025-01-03 | 2025-01-04 | src/feature.py                     | added       | 50          | 0             | 50                  |

#### 合并 PR 统计文件 (`gitee_prs_ascend_msit_hhhqx_stats_details.csv`)
| PR Number | PR Title                  | Source Files Changed | Total Lines Added | Total Lines Removed | Total Lines Changed | Days to Merge |
|-----------|---------------------------|-----------------------|-------------------|---------------------|---------------------|---------------|
| 1234      | Fix bug in module         | 1                     | 10                | 2                   | 12                  | 1.0           |
| 1235      | Add new feature           | 1                     | 50                | 0                   | 50                  | 1.0           |

#### 汇总统计文件 (`gitee_prs_ascend_msit_hhhqx_stats_summary.csv`)
| Total Merged PRs | Total Source Files Changed | Total Lines Added | Total Lines Removed | Total Lines Changed | Average Days to Merge | Median Days to Merge |
|-------------------|---------------------------|-------------------|---------------------|---------------------|-----------------------|----------------------|
| 2                 | 2                         | 60                | 2                   | 62                  | 1.0                   | 1.0                  |

## 参考资料
- Gitee API 参考页面：
  https://gitee.com/api/v5/swagger#/getV5ReposOwnerRepoPulls

