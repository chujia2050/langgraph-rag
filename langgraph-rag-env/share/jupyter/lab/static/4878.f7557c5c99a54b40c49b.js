"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[4878],{54878:(e,t,n)=>{n.r(t);n.d(t,{puppet:()=>c});var i={};var a=/({)?([a-z][a-z0-9_]*)?((::[a-z][a-z0-9_]*)*::)?[a-zA-Z0-9_]+(})?/;function r(e,t){var n=t.split(" ");for(var a=0;a<n.length;a++){i[n[a]]=e}}r("keyword","class define site node include import inherits");r("keyword","case if else in and elsif default or");r("atom","false true running present absent file directory undef");r("builtin","action augeas burst chain computer cron destination dport exec "+"file filebucket group host icmp iniface interface jump k5login limit log_level "+"log_prefix macauthorization mailalias maillist mcx mount nagios_command "+"nagios_contact nagios_contactgroup nagios_host nagios_hostdependency "+"nagios_hostescalation nagios_hostextinfo nagios_hostgroup nagios_service "+"nagios_servicedependency nagios_serviceescalation nagios_serviceextinfo "+"nagios_servicegroup nagios_timeperiod name notify outiface package proto reject "+"resources router schedule scheduled_task selboolean selmodule service source "+"sport ssh_authorized_key sshkey stage state table tidy todest toports tosource "+"user vlan yumrepo zfs zone zpool");function s(e,t){var n,i,a=false;while(!e.eol()&&(n=e.next())!=t.pending){if(n==="$"&&i!="\\"&&t.pending=='"'){a=true;break}i=n}if(a){e.backUp(1)}if(n==t.pending){t.continueString=false}else{t.continueString=true}return"string"}function o(e,t){var n=e.match(/[\w]+/,false);var r=e.match(/(\s+)?\w+\s+=>.*/,false);var o=e.match(/(\s+)?[\w:_]+(\s+)?{/,false);var c=e.match(/(\s+)?[@]{1,2}[\w:_]+(\s+)?{/,false);var u=e.next();if(u==="$"){if(e.match(a)){return t.continueString?"variableName.special":"variable"}return"error"}if(t.continueString){e.backUp(1);return s(e,t)}if(t.inDefinition){if(e.match(/(\s+)?[\w:_]+(\s+)?/)){return"def"}e.match(/\s+{/);t.inDefinition=false}if(t.inInclude){e.match(/(\s+)?\S+(\s+)?/);t.inInclude=false;return"def"}if(e.match(/(\s+)?\w+\(/)){e.backUp(1);return"def"}if(r){e.match(/(\s+)?\w+/);return"tag"}if(n&&i.hasOwnProperty(n)){e.backUp(1);e.match(/[\w]+/);if(e.match(/\s+\S+\s+{/,false)){t.inDefinition=true}if(n=="include"){t.inInclude=true}return i[n]}if(/(^|\s+)[A-Z][\w:_]+/.test(n)){e.backUp(1);e.match(/(^|\s+)[A-Z][\w:_]+/);return"def"}if(o){e.match(/(\s+)?[\w:_]+/);return"def"}if(c){e.match(/(\s+)?[@]{1,2}/);return"atom"}if(u=="#"){e.skipToEnd();return"comment"}if(u=="'"||u=='"'){t.pending=u;return s(e,t)}if(u=="{"||u=="}"){return"bracket"}if(u=="/"){e.match(/^[^\/]*\//);return"string.special"}if(u.match(/[0-9]/)){e.eatWhile(/[0-9]+/);return"number"}if(u=="="){if(e.peek()==">"){e.next()}return"operator"}e.eatWhile(/[\w-]/);return null}const c={name:"puppet",startState:function(){var e={};e.inDefinition=false;e.inInclude=false;e.continueString=false;e.pending=false;return e},token:function(e,t){if(e.eatSpace())return null;return o(e,t)}}}}]);