# frozen_string_literal: true

require "open3"

Jekyll::Hooks.register :posts, :post_init do |post|
  next if post.data["last_modified_at"]

  commit_count, count_status = Open3.capture2(
    "git", "rev-list", "--count", "HEAD", "--", post.path,
    chdir: post.site.source
  )
  next unless count_status.success? && commit_count.to_i > 1

  last_modified_at, log_status = Open3.capture2(
    "git", "log", "-1", "--format=%ad", "--date=iso", "--", post.path,
    chdir: post.site.source
  )
  post.data["last_modified_at"] = last_modified_at.strip if log_status.success?
end
