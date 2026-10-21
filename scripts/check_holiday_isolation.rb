# frozen_string_literal: true

# Build both variants at the same time and compare the actual published bytes.
# This covers HTML, search, sitemap, feed, JS and service-worker configuration.
require "jekyll"
require "tmpdir"
require "pathname"

source = File.expand_path("..", __dir__)
ENV["JEKYLL_ENV"] = "production"

Dir.mktmpdir("holiday-isolation-") do |temporary|
  config = Jekyll.configuration(
    "source" => source,
    "quiet" => true,
    "disable_disk_cache" => true,
    "incremental" => false,
    "time" => Time.now
  )
  outputs = %w[without-calendar with-calendar].map do |name|
    destination = File.join(temporary, name)
    settings = config.merge("destination" => destination)
    if name == "without-calendar"
      settings["exclude"] = config["exclude"] + ["assets/holidays"]
    end
    Jekyll::Site.new(settings).process
    Dir.glob(File.join(destination, "**", "*"), File::FNM_DOTMATCH)
       .select { |path| File.file?(path) }
       .to_h { |path| [Pathname.new(path).relative_path_from(Pathname.new(destination)).to_s, File.binread(path)] }
  end

  before, after = outputs
  expected = Dir.glob(File.join(source, "assets/holidays/cn/*.json"))
                .map { |path| Pathname.new(path).relative_path_from(Pathname.new(source)).to_s }
  raise "No holiday JSON files found" if expected.empty?
  raise "Unexpected published files added" unless (after.keys - before.keys).sort == expected.sort
  raise "Existing published files removed" unless (before.keys - after.keys).empty?

  changed = before.keys.select { |path| before[path] != after[path] }
  raise "Calendar data changed existing output: #{changed.join(', ')}" unless changed.empty?

  expected.each do |path|
    raise "Published JSON differs from source: #{path}" unless after[path] == File.binread(File.join(source, path))
  end
  forbidden = after.keys.grep(%r{\A(?:holidays(?:[./]|$)|_?docs/|README\.md$|\.harness/)})
  raise "Repository-only files or download pages were published: #{forbidden.join(', ')}" unless forbidden.empty?
  before.each do |path, bytes|
    next unless %w[.html .xml .json .js].include?(File.extname(path))
    if bytes.include?("/assets/holidays/") || bytes.include?("/holidays/")
      raise "Calendar endpoint leaked into blog output: #{path}"
    end
  end
  puts "PASS: #{before.size} existing published files byte-identical; only #{expected.size} static JSON files added"
  puts "PASS: no calendar links or download page; maintenance docs remain unpublished"
end
