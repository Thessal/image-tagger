extern crate tensorflow;


use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn tf_main() -> Result<(), Box<dyn Error>> {
    let filename = "examples/addition/model.pb"; // z = x + y
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python addition.py' to generate {} \
                     and try again.",
                    filename
                ),
            )
            .unwrap(),
        ));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[1]);
    x[0] = 2i32;
    let mut y = Tensor::new(&[1]);
    y[0] = 40i32;

    // Load the computation graph defined by addition.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("x")?, 0, &x);
    args.add_feed(&graph.operation_by_name_required("y")?, 0, &y);
    let z = args.request_fetch(&graph.operation_by_name_required("z")?, 0);
    session.run(&mut args)?;

    // Check our results.
    let z_res: i32 = args.fetch(z)?[0];
    println!("{:?}", z_res);

    Ok(())
}



use std::collections::HashMap;

use gloo_file::callbacks::FileReader;
use gloo_file::File;
use web_sys::{Event, HtmlInputElement};
use yew::html::TargetCast;
use yew::{html, Component, Context, Html};

type Chunks = bool;

pub enum Msg {
    Loaded(String, String),
    LoadedBytes(String, Vec<u8>),
    Files(Vec<File>, Chunks),
    ToggleReadBytes,
}

pub struct App {
    readers: HashMap<String, FileReader>,
    files: Vec<String>,
    read_bytes: bool,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            readers: HashMap::default(),
            files: vec![],
            read_bytes: false,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Loaded(file_name, data) => {
                let info = format!("file_name: {}, data: {:?}", file_name, data);
                self.files.push(info);
                self.readers.remove(&file_name);
                true
            }
            Msg::LoadedBytes(file_name, data) => {
                let info = format!("file_name: {}, data: {:?}", file_name, data);
                self.files.push(info);
                self.readers.remove(&file_name);
                true
            }
            Msg::Files(files, bytes) => {
                for file in files.into_iter() {
                    let file_name = file.name();
                    let task = {
                        let file_name = file_name.clone();
                        let link = ctx.link().clone();

                        if bytes {
                            gloo_file::callbacks::read_as_bytes(&file, move |res| {
                                link.send_message(Msg::LoadedBytes(
                                    file_name,
                                    res.expect("failed to read file"),
                                ))
                            })
                        } else {
                            gloo_file::callbacks::read_as_text(&file, move |res| {
                                link.send_message(Msg::Loaded(
                                    file_name,
                                    res.unwrap_or_else(|e| e.to_string()),
                                ))
                            })
                        }
                    };
                    self.readers.insert(file_name, task);
                }
                true
            }
            Msg::ToggleReadBytes => {
                self.read_bytes = !self.read_bytes;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let flag = self.read_bytes;
        html! {
            <div>
                <div>
                    <p>{ "Choose a file to upload to see the uploaded bytes" }</p>
                    <input type="file" multiple=true onchange={ctx.link().callback(move |e: Event| {
                            let mut result = Vec::new();
                            let input: HtmlInputElement = e.target_unchecked_into();

                            if let Some(files) = input.files() {
                                let files = js_sys::try_iter(&files)
                                    .unwrap()
                                    .unwrap()
                                    .map(|v| web_sys::File::from(v.unwrap()))
                                    .map(File::from);
                                result.extend(files);
                            }
                            Msg::Files(result, flag)
                        })}
                    />
                </div>
                <div>
                    <label>{ "Read bytes" }</label>
                    <input type="checkbox" checked={flag} onclick={ctx.link().callback(|_| Msg::ToggleReadBytes)} />
                </div>
                <ul>
                    { for self.files.iter().map(|f| Self::view_file(f)) }
                </ul>
            </div>
        }
    }
}

impl App {
    fn view_file(data: &str) -> Html {
        html! {
            <li>{ data }</li>
        }
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}