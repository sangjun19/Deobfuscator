	.file	"Par4All_par4all_bardin_thesis_2-03_flatten.c"
	.text
	.globl	_TIG_IZ_fpUC_argc
	.bss
	.align 4
	.type	_TIG_IZ_fpUC_argc, @object
	.size	_TIG_IZ_fpUC_argc, 4
_TIG_IZ_fpUC_argc:
	.zero	4
	.globl	_TIG_IZ_fpUC_argv
	.align 8
	.type	_TIG_IZ_fpUC_argv, @object
	.size	_TIG_IZ_fpUC_argv, 8
_TIG_IZ_fpUC_argv:
	.zero	8
	.globl	_TIG_IZ_fpUC_envp
	.align 8
	.type	_TIG_IZ_fpUC_envp, @object
	.size	_TIG_IZ_fpUC_envp, 8
_TIG_IZ_fpUC_envp:
	.zero	8
	.text
	.globl	ts_restructured
	.type	ts_restructured, @function
ts_restructured:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	$123, -8(%rbp)
.L713:
	cmpq	$444, -8(%rbp)
	ja	.L714
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L386-.L4
	.long	.L385-.L4
	.long	.L384-.L4
	.long	.L383-.L4
	.long	.L382-.L4
	.long	.L381-.L4
	.long	.L380-.L4
	.long	.L379-.L4
	.long	.L378-.L4
	.long	.L377-.L4
	.long	.L376-.L4
	.long	.L714-.L4
	.long	.L375-.L4
	.long	.L374-.L4
	.long	.L373-.L4
	.long	.L372-.L4
	.long	.L371-.L4
	.long	.L370-.L4
	.long	.L369-.L4
	.long	.L368-.L4
	.long	.L367-.L4
	.long	.L366-.L4
	.long	.L365-.L4
	.long	.L714-.L4
	.long	.L364-.L4
	.long	.L363-.L4
	.long	.L362-.L4
	.long	.L714-.L4
	.long	.L361-.L4
	.long	.L714-.L4
	.long	.L360-.L4
	.long	.L359-.L4
	.long	.L714-.L4
	.long	.L358-.L4
	.long	.L357-.L4
	.long	.L356-.L4
	.long	.L355-.L4
	.long	.L354-.L4
	.long	.L353-.L4
	.long	.L352-.L4
	.long	.L714-.L4
	.long	.L351-.L4
	.long	.L350-.L4
	.long	.L349-.L4
	.long	.L348-.L4
	.long	.L347-.L4
	.long	.L714-.L4
	.long	.L346-.L4
	.long	.L345-.L4
	.long	.L344-.L4
	.long	.L343-.L4
	.long	.L342-.L4
	.long	.L341-.L4
	.long	.L340-.L4
	.long	.L339-.L4
	.long	.L714-.L4
	.long	.L338-.L4
	.long	.L337-.L4
	.long	.L336-.L4
	.long	.L335-.L4
	.long	.L334-.L4
	.long	.L333-.L4
	.long	.L714-.L4
	.long	.L332-.L4
	.long	.L331-.L4
	.long	.L330-.L4
	.long	.L329-.L4
	.long	.L328-.L4
	.long	.L327-.L4
	.long	.L326-.L4
	.long	.L325-.L4
	.long	.L714-.L4
	.long	.L324-.L4
	.long	.L323-.L4
	.long	.L714-.L4
	.long	.L322-.L4
	.long	.L321-.L4
	.long	.L320-.L4
	.long	.L714-.L4
	.long	.L319-.L4
	.long	.L318-.L4
	.long	.L714-.L4
	.long	.L714-.L4
	.long	.L317-.L4
	.long	.L316-.L4
	.long	.L315-.L4
	.long	.L314-.L4
	.long	.L313-.L4
	.long	.L714-.L4
	.long	.L312-.L4
	.long	.L311-.L4
	.long	.L310-.L4
	.long	.L714-.L4
	.long	.L309-.L4
	.long	.L308-.L4
	.long	.L307-.L4
	.long	.L306-.L4
	.long	.L305-.L4
	.long	.L304-.L4
	.long	.L303-.L4
	.long	.L302-.L4
	.long	.L714-.L4
	.long	.L301-.L4
	.long	.L300-.L4
	.long	.L714-.L4
	.long	.L299-.L4
	.long	.L298-.L4
	.long	.L297-.L4
	.long	.L296-.L4
	.long	.L295-.L4
	.long	.L294-.L4
	.long	.L293-.L4
	.long	.L292-.L4
	.long	.L291-.L4
	.long	.L290-.L4
	.long	.L714-.L4
	.long	.L289-.L4
	.long	.L288-.L4
	.long	.L287-.L4
	.long	.L286-.L4
	.long	.L285-.L4
	.long	.L284-.L4
	.long	.L715-.L4
	.long	.L282-.L4
	.long	.L281-.L4
	.long	.L280-.L4
	.long	.L279-.L4
	.long	.L278-.L4
	.long	.L277-.L4
	.long	.L276-.L4
	.long	.L275-.L4
	.long	.L274-.L4
	.long	.L273-.L4
	.long	.L272-.L4
	.long	.L271-.L4
	.long	.L270-.L4
	.long	.L714-.L4
	.long	.L269-.L4
	.long	.L268-.L4
	.long	.L267-.L4
	.long	.L266-.L4
	.long	.L265-.L4
	.long	.L264-.L4
	.long	.L263-.L4
	.long	.L262-.L4
	.long	.L261-.L4
	.long	.L260-.L4
	.long	.L259-.L4
	.long	.L714-.L4
	.long	.L258-.L4
	.long	.L257-.L4
	.long	.L256-.L4
	.long	.L714-.L4
	.long	.L255-.L4
	.long	.L254-.L4
	.long	.L253-.L4
	.long	.L252-.L4
	.long	.L251-.L4
	.long	.L250-.L4
	.long	.L249-.L4
	.long	.L248-.L4
	.long	.L247-.L4
	.long	.L246-.L4
	.long	.L245-.L4
	.long	.L244-.L4
	.long	.L243-.L4
	.long	.L242-.L4
	.long	.L241-.L4
	.long	.L240-.L4
	.long	.L239-.L4
	.long	.L238-.L4
	.long	.L237-.L4
	.long	.L236-.L4
	.long	.L714-.L4
	.long	.L235-.L4
	.long	.L234-.L4
	.long	.L233-.L4
	.long	.L232-.L4
	.long	.L231-.L4
	.long	.L230-.L4
	.long	.L714-.L4
	.long	.L229-.L4
	.long	.L228-.L4
	.long	.L714-.L4
	.long	.L714-.L4
	.long	.L227-.L4
	.long	.L226-.L4
	.long	.L225-.L4
	.long	.L224-.L4
	.long	.L223-.L4
	.long	.L222-.L4
	.long	.L221-.L4
	.long	.L220-.L4
	.long	.L219-.L4
	.long	.L218-.L4
	.long	.L217-.L4
	.long	.L216-.L4
	.long	.L215-.L4
	.long	.L214-.L4
	.long	.L213-.L4
	.long	.L212-.L4
	.long	.L211-.L4
	.long	.L210-.L4
	.long	.L209-.L4
	.long	.L714-.L4
	.long	.L208-.L4
	.long	.L207-.L4
	.long	.L206-.L4
	.long	.L205-.L4
	.long	.L204-.L4
	.long	.L203-.L4
	.long	.L202-.L4
	.long	.L201-.L4
	.long	.L200-.L4
	.long	.L199-.L4
	.long	.L714-.L4
	.long	.L714-.L4
	.long	.L198-.L4
	.long	.L197-.L4
	.long	.L714-.L4
	.long	.L196-.L4
	.long	.L195-.L4
	.long	.L194-.L4
	.long	.L193-.L4
	.long	.L192-.L4
	.long	.L191-.L4
	.long	.L190-.L4
	.long	.L189-.L4
	.long	.L188-.L4
	.long	.L187-.L4
	.long	.L186-.L4
	.long	.L185-.L4
	.long	.L184-.L4
	.long	.L183-.L4
	.long	.L182-.L4
	.long	.L714-.L4
	.long	.L181-.L4
	.long	.L180-.L4
	.long	.L179-.L4
	.long	.L178-.L4
	.long	.L714-.L4
	.long	.L177-.L4
	.long	.L176-.L4
	.long	.L175-.L4
	.long	.L174-.L4
	.long	.L173-.L4
	.long	.L172-.L4
	.long	.L171-.L4
	.long	.L170-.L4
	.long	.L714-.L4
	.long	.L169-.L4
	.long	.L168-.L4
	.long	.L167-.L4
	.long	.L166-.L4
	.long	.L165-.L4
	.long	.L164-.L4
	.long	.L163-.L4
	.long	.L162-.L4
	.long	.L161-.L4
	.long	.L160-.L4
	.long	.L159-.L4
	.long	.L158-.L4
	.long	.L157-.L4
	.long	.L156-.L4
	.long	.L155-.L4
	.long	.L154-.L4
	.long	.L153-.L4
	.long	.L152-.L4
	.long	.L151-.L4
	.long	.L150-.L4
	.long	.L149-.L4
	.long	.L148-.L4
	.long	.L147-.L4
	.long	.L146-.L4
	.long	.L145-.L4
	.long	.L144-.L4
	.long	.L143-.L4
	.long	.L142-.L4
	.long	.L141-.L4
	.long	.L140-.L4
	.long	.L139-.L4
	.long	.L138-.L4
	.long	.L137-.L4
	.long	.L136-.L4
	.long	.L135-.L4
	.long	.L134-.L4
	.long	.L714-.L4
	.long	.L133-.L4
	.long	.L132-.L4
	.long	.L714-.L4
	.long	.L131-.L4
	.long	.L130-.L4
	.long	.L129-.L4
	.long	.L128-.L4
	.long	.L127-.L4
	.long	.L126-.L4
	.long	.L125-.L4
	.long	.L124-.L4
	.long	.L123-.L4
	.long	.L122-.L4
	.long	.L121-.L4
	.long	.L120-.L4
	.long	.L119-.L4
	.long	.L118-.L4
	.long	.L117-.L4
	.long	.L116-.L4
	.long	.L115-.L4
	.long	.L114-.L4
	.long	.L113-.L4
	.long	.L112-.L4
	.long	.L111-.L4
	.long	.L110-.L4
	.long	.L714-.L4
	.long	.L109-.L4
	.long	.L108-.L4
	.long	.L107-.L4
	.long	.L106-.L4
	.long	.L714-.L4
	.long	.L105-.L4
	.long	.L714-.L4
	.long	.L104-.L4
	.long	.L103-.L4
	.long	.L102-.L4
	.long	.L101-.L4
	.long	.L100-.L4
	.long	.L714-.L4
	.long	.L99-.L4
	.long	.L98-.L4
	.long	.L97-.L4
	.long	.L96-.L4
	.long	.L95-.L4
	.long	.L94-.L4
	.long	.L93-.L4
	.long	.L92-.L4
	.long	.L91-.L4
	.long	.L90-.L4
	.long	.L714-.L4
	.long	.L714-.L4
	.long	.L89-.L4
	.long	.L88-.L4
	.long	.L87-.L4
	.long	.L86-.L4
	.long	.L85-.L4
	.long	.L84-.L4
	.long	.L83-.L4
	.long	.L82-.L4
	.long	.L714-.L4
	.long	.L81-.L4
	.long	.L714-.L4
	.long	.L714-.L4
	.long	.L80-.L4
	.long	.L79-.L4
	.long	.L78-.L4
	.long	.L77-.L4
	.long	.L76-.L4
	.long	.L75-.L4
	.long	.L74-.L4
	.long	.L73-.L4
	.long	.L72-.L4
	.long	.L71-.L4
	.long	.L70-.L4
	.long	.L714-.L4
	.long	.L69-.L4
	.long	.L68-.L4
	.long	.L67-.L4
	.long	.L66-.L4
	.long	.L714-.L4
	.long	.L65-.L4
	.long	.L64-.L4
	.long	.L63-.L4
	.long	.L714-.L4
	.long	.L62-.L4
	.long	.L714-.L4
	.long	.L61-.L4
	.long	.L60-.L4
	.long	.L59-.L4
	.long	.L714-.L4
	.long	.L714-.L4
	.long	.L58-.L4
	.long	.L714-.L4
	.long	.L57-.L4
	.long	.L714-.L4
	.long	.L56-.L4
	.long	.L55-.L4
	.long	.L54-.L4
	.long	.L53-.L4
	.long	.L52-.L4
	.long	.L714-.L4
	.long	.L51-.L4
	.long	.L714-.L4
	.long	.L50-.L4
	.long	.L49-.L4
	.long	.L48-.L4
	.long	.L47-.L4
	.long	.L46-.L4
	.long	.L714-.L4
	.long	.L714-.L4
	.long	.L45-.L4
	.long	.L714-.L4
	.long	.L44-.L4
	.long	.L43-.L4
	.long	.L42-.L4
	.long	.L41-.L4
	.long	.L714-.L4
	.long	.L40-.L4
	.long	.L39-.L4
	.long	.L38-.L4
	.long	.L37-.L4
	.long	.L36-.L4
	.long	.L35-.L4
	.long	.L34-.L4
	.long	.L33-.L4
	.long	.L32-.L4
	.long	.L31-.L4
	.long	.L30-.L4
	.long	.L714-.L4
	.long	.L29-.L4
	.long	.L28-.L4
	.long	.L27-.L4
	.long	.L26-.L4
	.long	.L25-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L714-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L714-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L714-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L178:
	call	deadlock
	movq	$417, -8(%rbp)
	jmp	.L387
.L52:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$77, -8(%rbp)
	jmp	.L387
.L228:
	cmpl	$0, -120(%rbp)
	je	.L388
	movq	$354, -8(%rbp)
	jmp	.L387
.L388:
	movq	$238, -8(%rbp)
	jmp	.L387
.L105:
	call	deadlock
	movq	$343, -8(%rbp)
	jmp	.L387
.L19:
	movl	$2, -120(%rbp)
	movq	$288, -8(%rbp)
	jmp	.L387
.L369:
	cmpl	$0, -116(%rbp)
	jns	.L390
	movq	$242, -8(%rbp)
	jmp	.L387
.L390:
	movq	$114, -8(%rbp)
	jmp	.L387
.L343:
	cmpl	$0, -108(%rbp)
	je	.L392
	movq	$26, -8(%rbp)
	jmp	.L387
.L392:
	movq	$375, -8(%rbp)
	jmp	.L387
.L318:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$433, -8(%rbp)
	jmp	.L387
.L120:
	cmpl	$2, -120(%rbp)
	je	.L394
	movq	$2, -8(%rbp)
	jmp	.L387
.L394:
	movq	$138, -8(%rbp)
	jmp	.L387
.L119:
	movl	$2, -120(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L387
.L76:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	js	.L396
	movq	$382, -8(%rbp)
	jmp	.L387
.L396:
	movq	$241, -8(%rbp)
	jmp	.L387
.L224:
	call	rand_b
	movl	%eax, -32(%rbp)
	movq	$294, -8(%rbp)
	jmp	.L387
.L382:
	cmpl	$0, -120(%rbp)
	je	.L398
	movq	$363, -8(%rbp)
	jmp	.L387
.L398:
	movq	$228, -8(%rbp)
	jmp	.L387
.L161:
	cmpl	$1, -120(%rbp)
	jne	.L400
	movq	$163, -8(%rbp)
	jmp	.L387
.L400:
	movq	$267, -8(%rbp)
	jmp	.L387
.L300:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L402
	movq	$335, -8(%rbp)
	jmp	.L387
.L402:
	movq	$185, -8(%rbp)
	jmp	.L387
.L223:
	cmpl	$0, -116(%rbp)
	js	.L404
	movq	$172, -8(%rbp)
	jmp	.L387
.L404:
	movq	$135, -8(%rbp)
	jmp	.L387
.L301:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L406
	movq	$260, -8(%rbp)
	jmp	.L387
.L406:
	movq	$116, -8(%rbp)
	jmp	.L387
.L147:
	cmpl	$0, -116(%rbp)
	jns	.L408
	movq	$156, -8(%rbp)
	jmp	.L387
.L408:
	movq	$197, -8(%rbp)
	jmp	.L387
.L261:
	call	deadlock
	movq	$315, -8(%rbp)
	jmp	.L387
.L58:
	call	checking_error
	movq	$414, -8(%rbp)
	jmp	.L387
.L3:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L410
	movq	$417, -8(%rbp)
	jmp	.L387
.L410:
	movq	$199, -8(%rbp)
	jmp	.L387
.L9:
	call	rand_b
	movl	%eax, -40(%rbp)
	movq	$205, -8(%rbp)
	jmp	.L387
.L372:
	call	checking_error
	movq	$439, -8(%rbp)
	jmp	.L387
.L195:
	call	checking_error
	movq	$197, -8(%rbp)
	jmp	.L387
.L144:
	call	deadlock
	movq	$208, -8(%rbp)
	jmp	.L387
.L43:
	call	deadlock
	movq	$313, -8(%rbp)
	jmp	.L387
.L18:
	call	deadlock
	movq	$208, -8(%rbp)
	jmp	.L387
.L284:
	call	rand_b
	movl	%eax, -44(%rbp)
	movq	$364, -8(%rbp)
	jmp	.L387
.L265:
	call	deadlock
	movq	$261, -8(%rbp)
	jmp	.L387
.L30:
	call	rand_b
	movl	%eax, -60(%rbp)
	movq	$298, -8(%rbp)
	jmp	.L387
.L240:
	cmpl	$0, -116(%rbp)
	js	.L412
	movq	$126, -8(%rbp)
	jmp	.L387
.L412:
	movq	$442, -8(%rbp)
	jmp	.L387
.L208:
	cmpl	$0, -40(%rbp)
	je	.L414
	movq	$307, -8(%rbp)
	jmp	.L387
.L414:
	movq	$140, -8(%rbp)
	jmp	.L387
.L53:
	cmpl	$0, -120(%rbp)
	je	.L416
	movq	$269, -8(%rbp)
	jmp	.L387
.L416:
	movq	$165, -8(%rbp)
	jmp	.L387
.L378:
	cmpl	$1, -120(%rbp)
	jne	.L418
	movq	$244, -8(%rbp)
	jmp	.L387
.L418:
	movq	$412, -8(%rbp)
	jmp	.L387
.L248:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$97, -8(%rbp)
	jmp	.L387
.L172:
	call	deadlock
	movq	$390, -8(%rbp)
	jmp	.L387
.L17:
	cmpl	$1, -120(%rbp)
	jne	.L420
	movq	$9, -8(%rbp)
	jmp	.L387
.L420:
	movq	$108, -8(%rbp)
	jmp	.L387
.L90:
	movl	$2, -120(%rbp)
	movq	$217, -8(%rbp)
	jmp	.L387
.L20:
	cmpl	$0, -116(%rbp)
	jns	.L422
	movq	$51, -8(%rbp)
	jmp	.L387
.L422:
	movq	$121, -8(%rbp)
	jmp	.L387
.L11:
	call	deadlock
	movq	$315, -8(%rbp)
	jmp	.L387
.L217:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L424
	movq	$99, -8(%rbp)
	jmp	.L387
.L424:
	movq	$437, -8(%rbp)
	jmp	.L387
.L339:
	call	deadlock
	movq	$232, -8(%rbp)
	jmp	.L387
.L116:
	call	deadlock
	movq	$408, -8(%rbp)
	jmp	.L387
.L281:
	cmpl	$2, -120(%rbp)
	je	.L426
	movq	$153, -8(%rbp)
	jmp	.L387
.L426:
	movq	$186, -8(%rbp)
	jmp	.L387
.L266:
	movl	$2, -120(%rbp)
	movq	$35, -8(%rbp)
	jmp	.L387
.L385:
	cmpl	$0, -116(%rbp)
	jns	.L428
	movq	$24, -8(%rbp)
	jmp	.L387
.L428:
	movq	$421, -8(%rbp)
	jmp	.L387
.L320:
	cmpl	$0, -116(%rbp)
	jns	.L430
	movq	$266, -8(%rbp)
	jmp	.L387
.L430:
	movq	$125, -8(%rbp)
	jmp	.L387
.L176:
	call	checking_error
	movq	$114, -8(%rbp)
	jmp	.L387
.L93:
	call	deadlock
	movq	$113, -8(%rbp)
	jmp	.L387
.L84:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$402, -8(%rbp)
	jmp	.L387
.L249:
	call	deadlock
	movq	$89, -8(%rbp)
	jmp	.L387
.L383:
	cmpl	$0, -116(%rbp)
	jns	.L432
	movq	$91, -8(%rbp)
	jmp	.L387
.L432:
	movq	$256, -8(%rbp)
	jmp	.L387
.L26:
	call	deadlock
	movq	$335, -8(%rbp)
	jmp	.L387
.L371:
	cmpl	$0, -84(%rbp)
	je	.L434
	movq	$155, -8(%rbp)
	jmp	.L387
.L434:
	movq	$22, -8(%rbp)
	jmp	.L387
.L366:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$440, -8(%rbp)
	jmp	.L387
.L355:
	call	deadlock
	movq	$226, -8(%rbp)
	jmp	.L387
.L274:
	call	checking_error
	movq	$223, -8(%rbp)
	jmp	.L387
.L185:
	call	deadlock
	movq	$260, -8(%rbp)
	jmp	.L387
.L327:
	call	deadlock
	movq	$343, -8(%rbp)
	jmp	.L387
.L200:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L436
	movq	$102, -8(%rbp)
	jmp	.L387
.L436:
	movq	$194, -8(%rbp)
	jmp	.L387
.L101:
	cmpl	$0, -116(%rbp)
	jns	.L438
	movq	$328, -8(%rbp)
	jmp	.L387
.L438:
	movq	$439, -8(%rbp)
	jmp	.L387
.L229:
	call	checking_error
	movq	$49, -8(%rbp)
	jmp	.L387
.L180:
	call	checking_error
	movq	$223, -8(%rbp)
	jmp	.L387
.L115:
	cmpl	$0, -116(%rbp)
	jns	.L441
	movq	$279, -8(%rbp)
	jmp	.L387
.L441:
	movq	$253, -8(%rbp)
	jmp	.L387
.L56:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L443
	movq	$89, -8(%rbp)
	jmp	.L387
.L443:
	movq	$159, -8(%rbp)
	jmp	.L387
.L315:
	call	deadlock
	movq	$322, -8(%rbp)
	jmp	.L387
.L302:
	addl	$2, -116(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L387
.L41:
	cmpl	$0, -116(%rbp)
	jns	.L445
	movq	$283, -8(%rbp)
	jmp	.L387
.L445:
	movq	$256, -8(%rbp)
	jmp	.L387
.L362:
	cmpl	$1, -120(%rbp)
	jne	.L447
	movq	$436, -8(%rbp)
	jmp	.L387
.L447:
	movq	$63, -8(%rbp)
	jmp	.L387
.L299:
	movl	-116(%rbp), %eax
	addl	$2, %eax
	cmpl	%eax, -112(%rbp)
	jg	.L449
	movq	$408, -8(%rbp)
	jmp	.L387
.L449:
	movq	$106, -8(%rbp)
	jmp	.L387
.L32:
	call	deadlock
	movq	$347, -8(%rbp)
	jmp	.L387
.L125:
	movl	$2, -120(%rbp)
	movq	$385, -8(%rbp)
	jmp	.L387
.L13:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L451
	movq	$276, -8(%rbp)
	jmp	.L387
.L451:
	movq	$320, -8(%rbp)
	jmp	.L387
.L191:
	call	deadlock
	movq	$208, -8(%rbp)
	jmp	.L387
.L171:
	movl	$2, -120(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L387
.L270:
	call	deadlock
	movq	$271, -8(%rbp)
	jmp	.L387
.L332:
	call	deadlock
	movq	$193, -8(%rbp)
	jmp	.L387
.L140:
	call	checking_error
	movq	$253, -8(%rbp)
	jmp	.L387
.L114:
	cmpl	$2, -120(%rbp)
	je	.L453
	movq	$109, -8(%rbp)
	jmp	.L387
.L453:
	movq	$321, -8(%rbp)
	jmp	.L387
.L233:
	call	deadlock
	movq	$261, -8(%rbp)
	jmp	.L387
.L226:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$394, -8(%rbp)
	jmp	.L387
.L73:
	call	deadlock
	movq	$80, -8(%rbp)
	jmp	.L387
.L124:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L455
	movq	$192, -8(%rbp)
	jmp	.L387
.L455:
	movq	$280, -8(%rbp)
	jmp	.L387
.L220:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L457
	movq	$80, -8(%rbp)
	jmp	.L387
.L457:
	movq	$292, -8(%rbp)
	jmp	.L387
.L202:
	call	deadlock
	movq	$45, -8(%rbp)
	jmp	.L387
.L14:
	cmpl	$0, -116(%rbp)
	jns	.L459
	movq	$268, -8(%rbp)
	jmp	.L387
.L459:
	movq	$210, -8(%rbp)
	jmp	.L387
.L160:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L461
	movq	$427, -8(%rbp)
	jmp	.L387
.L461:
	movq	$149, -8(%rbp)
	jmp	.L387
.L148:
	addl	$2, -116(%rbp)
	movq	$119, -8(%rbp)
	jmp	.L387
.L280:
	call	rand_b
	movl	%eax, -80(%rbp)
	movq	$316, -8(%rbp)
	jmp	.L387
.L104:
	call	deadlock
	movq	$313, -8(%rbp)
	jmp	.L387
.L263:
	cmpl	$0, -116(%rbp)
	jns	.L463
	movq	$420, -8(%rbp)
	jmp	.L387
.L463:
	movq	$341, -8(%rbp)
	jmp	.L387
.L99:
	movl	$2, -120(%rbp)
	movq	$245, -8(%rbp)
	jmp	.L387
.L48:
	call	checking_error
	movq	$257, -8(%rbp)
	jmp	.L387
.L16:
	cmpl	$0, -64(%rbp)
	je	.L465
	movq	$200, -8(%rbp)
	jmp	.L387
.L465:
	movq	$326, -8(%rbp)
	jmp	.L387
.L368:
	call	checking_error
	movq	$53, -8(%rbp)
	jmp	.L387
.L189:
	call	deadlock
	movq	$247, -8(%rbp)
	jmp	.L387
.L370:
	call	deadlock
	movq	$386, -8(%rbp)
	jmp	.L387
.L236:
	movl	-116(%rbp), %eax
	addl	$2, %eax
	cmpl	%eax, -112(%rbp)
	jg	.L467
	movq	$271, -8(%rbp)
	jmp	.L387
.L467:
	movq	$42, -8(%rbp)
	jmp	.L387
.L188:
	call	deadlock
	movq	$150, -8(%rbp)
	jmp	.L387
.L49:
	cmpl	$0, -16(%rbp)
	je	.L469
	movq	$416, -8(%rbp)
	jmp	.L387
.L469:
	movq	$128, -8(%rbp)
	jmp	.L387
.L190:
	movl	$2, -120(%rbp)
	movq	$304, -8(%rbp)
	jmp	.L387
.L109:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$107, -8(%rbp)
	jmp	.L387
.L213:
	call	deadlock
	movq	$417, -8(%rbp)
	jmp	.L387
.L155:
	cmpl	$0, -116(%rbp)
	jns	.L471
	movq	$410, -8(%rbp)
	jmp	.L387
.L471:
	movq	$121, -8(%rbp)
	jmp	.L387
.L150:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L473
	movq	$355, -8(%rbp)
	jmp	.L387
.L473:
	movq	$374, -8(%rbp)
	jmp	.L387
.L126:
	call	checking_error
	movq	$309, -8(%rbp)
	jmp	.L387
.L288:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L475
	movq	$247, -8(%rbp)
	jmp	.L387
.L475:
	movq	$227, -8(%rbp)
	jmp	.L387
.L205:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$369, -8(%rbp)
	jmp	.L387
.L97:
	call	checking_error
	movq	$439, -8(%rbp)
	jmp	.L387
.L334:
	call	deadlock
	movq	$232, -8(%rbp)
	jmp	.L387
.L257:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$426, -8(%rbp)
	jmp	.L387
.L77:
	call	checking_error
	movq	$422, -8(%rbp)
	jmp	.L387
.L225:
	call	checking_error
	movq	$256, -8(%rbp)
	jmp	.L387
.L174:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L477
	movq	$347, -8(%rbp)
	jmp	.L387
.L477:
	movq	$137, -8(%rbp)
	jmp	.L387
.L247:
	cmpl	$0, -120(%rbp)
	je	.L479
	movq	$278, -8(%rbp)
	jmp	.L387
.L479:
	movq	$263, -8(%rbp)
	jmp	.L387
.L39:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$359, -8(%rbp)
	jmp	.L387
.L40:
	cmpl	$0, -72(%rbp)
	je	.L481
	movq	$301, -8(%rbp)
	jmp	.L387
.L481:
	movq	$338, -8(%rbp)
	jmp	.L387
.L264:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L483
	movq	$343, -8(%rbp)
	jmp	.L387
.L483:
	movq	$68, -8(%rbp)
	jmp	.L387
.L262:
	movl	-116(%rbp), %eax
	addl	$2, %eax
	cmpl	%eax, -112(%rbp)
	jle	.L485
	movq	$224, -8(%rbp)
	jmp	.L387
.L485:
	movq	$110, -8(%rbp)
	jmp	.L387
.L29:
	cmpl	$1, -120(%rbp)
	jne	.L487
	movq	$334, -8(%rbp)
	jmp	.L387
.L487:
	movq	$282, -8(%rbp)
	jmp	.L387
.L269:
	call	deadlock
	movq	$347, -8(%rbp)
	jmp	.L387
.L157:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jge	.L489
	movq	$314, -8(%rbp)
	jmp	.L387
.L489:
	movq	$256, -8(%rbp)
	jmp	.L387
.L175:
	movl	$2, -120(%rbp)
	movq	$383, -8(%rbp)
	jmp	.L387
.L251:
	cmpl	$0, -116(%rbp)
	jns	.L491
	movq	$293, -8(%rbp)
	jmp	.L387
.L491:
	movq	$314, -8(%rbp)
	jmp	.L387
.L218:
	call	deadlock
	movq	$260, -8(%rbp)
	jmp	.L387
.L91:
	cmpl	$0, -116(%rbp)
	js	.L493
	movq	$67, -8(%rbp)
	jmp	.L387
.L493:
	movq	$393, -8(%rbp)
	jmp	.L387
.L201:
	cmpl	$0, -116(%rbp)
	jns	.L495
	movq	$295, -8(%rbp)
	jmp	.L387
.L495:
	movq	$309, -8(%rbp)
	jmp	.L387
.L313:
	cmpl	$0, -116(%rbp)
	jns	.L497
	movq	$237, -8(%rbp)
	jmp	.L387
.L497:
	movq	$223, -8(%rbp)
	jmp	.L387
.L316:
	cmpl	$0, -116(%rbp)
	jns	.L499
	movq	$308, -8(%rbp)
	jmp	.L387
.L499:
	movq	$125, -8(%rbp)
	jmp	.L387
.L237:
	call	deadlock
	movq	$37, -8(%rbp)
	jmp	.L387
.L357:
	cmpl	$1, -120(%rbp)
	jne	.L501
	movq	$58, -8(%rbp)
	jmp	.L387
.L501:
	movq	$39, -8(%rbp)
	jmp	.L387
.L145:
	call	checking_error
	movq	$309, -8(%rbp)
	jmp	.L387
.L67:
	cmpl	$0, -44(%rbp)
	je	.L503
	movq	$14, -8(%rbp)
	jmp	.L387
.L503:
	movq	$330, -8(%rbp)
	jmp	.L387
.L27:
	movl	$2, -120(%rbp)
	movq	$151, -8(%rbp)
	jmp	.L387
.L177:
	call	deadlock
	movq	$89, -8(%rbp)
	jmp	.L387
.L168:
	call	deadlock
	movq	$271, -8(%rbp)
	jmp	.L387
.L95:
	cmpl	$1, -120(%rbp)
	jne	.L505
	movq	$438, -8(%rbp)
	jmp	.L387
.L505:
	movq	$60, -8(%rbp)
	jmp	.L387
.L15:
	cmpl	$0, -116(%rbp)
	jns	.L507
	movq	$13, -8(%rbp)
	jmp	.L387
.L507:
	movq	$341, -8(%rbp)
	jmp	.L387
.L239:
	call	deadlock
	movq	$250, -8(%rbp)
	jmp	.L387
.L138:
	cmpl	$0, -52(%rbp)
	je	.L509
	movq	$59, -8(%rbp)
	jmp	.L387
.L509:
	movq	$302, -8(%rbp)
	jmp	.L387
.L278:
	call	rand_b
	movl	%eax, -36(%rbp)
	movq	$48, -8(%rbp)
	jmp	.L387
.L71:
	cmpl	$0, -116(%rbp)
	jns	.L511
	movq	$0, -8(%rbp)
	jmp	.L387
.L511:
	movq	$114, -8(%rbp)
	jmp	.L387
.L31:
	call	checking_error
	movq	$53, -8(%rbp)
	jmp	.L387
.L365:
	movl	$2, -120(%rbp)
	movq	$179, -8(%rbp)
	jmp	.L387
.L194:
	call	deadlock
	movq	$224, -8(%rbp)
	jmp	.L387
.L123:
	cmpl	$0, -60(%rbp)
	je	.L513
	movq	$34, -8(%rbp)
	jmp	.L387
.L513:
	movq	$112, -8(%rbp)
	jmp	.L387
.L75:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L515
	movq	$61, -8(%rbp)
	jmp	.L387
.L515:
	movq	$214, -8(%rbp)
	jmp	.L387
.L361:
	call	deadlock
	movq	$386, -8(%rbp)
	jmp	.L387
.L7:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$380, -8(%rbp)
	jmp	.L387
.L330:
	cmpl	$1, -120(%rbp)
	jne	.L517
	movq	$30, -8(%rbp)
	jmp	.L387
.L517:
	movq	$350, -8(%rbp)
	jmp	.L387
.L139:
	call	deadlock
	movq	$80, -8(%rbp)
	jmp	.L387
.L8:
	cmpl	$0, -116(%rbp)
	jns	.L519
	movq	$94, -8(%rbp)
	jmp	.L387
.L519:
	movq	$273, -8(%rbp)
	jmp	.L387
.L348:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$272, -8(%rbp)
	jmp	.L387
.L22:
	cmpl	$0, -20(%rbp)
	je	.L521
	movq	$10, -8(%rbp)
	jmp	.L387
.L521:
	movq	$203, -8(%rbp)
	jmp	.L387
.L381:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L523
	movq	$261, -8(%rbp)
	jmp	.L387
.L523:
	movq	$176, -8(%rbp)
	jmp	.L387
.L285:
	call	deadlock
	movq	$250, -8(%rbp)
	jmp	.L387
.L66:
	cmpl	$0, -24(%rbp)
	je	.L525
	movq	$303, -8(%rbp)
	jmp	.L387
.L525:
	movq	$177, -8(%rbp)
	jmp	.L387
.L21:
	cmpl	$0, -76(%rbp)
	je	.L527
	movq	$360, -8(%rbp)
	jmp	.L387
.L527:
	movq	$285, -8(%rbp)
	jmp	.L387
.L246:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L529
	movq	$322, -8(%rbp)
	jmp	.L387
.L529:
	movq	$399, -8(%rbp)
	jmp	.L387
.L324:
	cmpl	$0, -116(%rbp)
	jns	.L531
	movq	$206, -8(%rbp)
	jmp	.L387
.L531:
	movq	$270, -8(%rbp)
	jmp	.L387
.L158:
	movl	$2, -120(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L387
.L358:
	call	deadlock
	movq	$243, -8(%rbp)
	jmp	.L387
.L331:
	movl	$2, -120(%rbp)
	movq	$147, -8(%rbp)
	jmp	.L387
.L46:
	cmpl	$0, -116(%rbp)
	jns	.L533
	movq	$19, -8(%rbp)
	jmp	.L387
.L533:
	movq	$53, -8(%rbp)
	jmp	.L387
.L129:
	call	deadlock
	movq	$80, -8(%rbp)
	jmp	.L387
.L309:
	cmpl	$0, -120(%rbp)
	je	.L535
	movq	$213, -8(%rbp)
	jmp	.L387
.L535:
	movq	$231, -8(%rbp)
	jmp	.L387
.L286:
	cmpl	$0, -116(%rbp)
	jns	.L537
	movq	$178, -8(%rbp)
	jmp	.L387
.L537:
	movq	$256, -8(%rbp)
	jmp	.L387
.L234:
	call	checking_error
	movq	$49, -8(%rbp)
	jmp	.L387
.L89:
	movl	$2, -120(%rbp)
	movq	$161, -8(%rbp)
	jmp	.L387
.L143:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jle	.L539
	movq	$313, -8(%rbp)
	jmp	.L387
.L539:
	movq	$400, -8(%rbp)
	jmp	.L387
.L183:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L541
	movq	$190, -8(%rbp)
	jmp	.L387
.L541:
	movq	$191, -8(%rbp)
	jmp	.L387
.L112:
	call	rand_b
	movl	%eax, -72(%rbp)
	movq	$404, -8(%rbp)
	jmp	.L387
.L376:
	cmpl	$1, -120(%rbp)
	jne	.L543
	movq	$5, -8(%rbp)
	jmp	.L387
.L543:
	movq	$141, -8(%rbp)
	jmp	.L387
.L386:
	call	checking_error
	movq	$114, -8(%rbp)
	jmp	.L387
.L192:
	addl	$2, -116(%rbp)
	movq	$388, -8(%rbp)
	jmp	.L387
.L165:
	call	checking_error
	movq	$253, -8(%rbp)
	jmp	.L387
.L28:
	movl	$2, -120(%rbp)
	movq	$98, -8(%rbp)
	jmp	.L387
.L255:
	call	deadlock
	movq	$186, -8(%rbp)
	jmp	.L387
.L352:
	call	deadlock
	movq	$224, -8(%rbp)
	jmp	.L387
.L206:
	call	deadlock
	movq	$193, -8(%rbp)
	jmp	.L387
.L379:
	call	deadlock
	movq	$427, -8(%rbp)
	jmp	.L387
.L235:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jle	.L545
	movq	$208, -8(%rbp)
	jmp	.L387
.L545:
	movq	$428, -8(%rbp)
	jmp	.L387
.L65:
	cmpl	$1, -120(%rbp)
	jne	.L547
	movq	$90, -8(%rbp)
	jmp	.L387
.L547:
	movq	$166, -8(%rbp)
	jmp	.L387
.L279:
	movl	-116(%rbp), %eax
	addl	$2, %eax
	cmpl	%eax, -112(%rbp)
	jle	.L549
	movq	$100, -8(%rbp)
	jmp	.L387
.L549:
	movq	$69, -8(%rbp)
	jmp	.L387
.L230:
	cmpl	$0, -120(%rbp)
	je	.L551
	movq	$277, -8(%rbp)
	jmp	.L387
.L551:
	movq	$28, -8(%rbp)
	jmp	.L387
.L96:
	cmpl	$0, -116(%rbp)
	js	.L553
	movq	$105, -8(%rbp)
	jmp	.L387
.L553:
	movq	$305, -8(%rbp)
	jmp	.L387
.L80:
	call	deadlock
	movq	$373, -8(%rbp)
	jmp	.L387
.L367:
	call	checking_error
	movq	$421, -8(%rbp)
	jmp	.L387
.L107:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$143, -8(%rbp)
	jmp	.L387
.L127:
	cmpl	$0, -32(%rbp)
	je	.L555
	movq	$258, -8(%rbp)
	jmp	.L387
.L555:
	movq	$127, -8(%rbp)
	jmp	.L387
.L78:
	cmpl	$0, -88(%rbp)
	je	.L557
	movq	$158, -8(%rbp)
	jmp	.L387
.L557:
	movq	$130, -8(%rbp)
	jmp	.L387
.L35:
	cmpl	$0, -100(%rbp)
	je	.L559
	movq	$299, -8(%rbp)
	jmp	.L387
.L559:
	movq	$296, -8(%rbp)
	jmp	.L387
.L276:
	movl	$2, -120(%rbp)
	movq	$93, -8(%rbp)
	jmp	.L387
.L45:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$201, -8(%rbp)
	jmp	.L387
.L296:
	call	deadlock
	movq	$226, -8(%rbp)
	jmp	.L387
.L242:
	call	deadlock
	movq	$243, -8(%rbp)
	jmp	.L387
.L72:
	movl	$1, -120(%rbp)
	call	rand@PLT
	movl	%eax, -116(%rbp)
	call	rand@PLT
	movl	%eax, -112(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L387
.L275:
	movl	$2, -120(%rbp)
	movq	$182, -8(%rbp)
	jmp	.L387
.L111:
	call	deadlock
	movq	$373, -8(%rbp)
	jmp	.L387
.L62:
	cmpl	$0, -116(%rbp)
	jns	.L561
	movq	$187, -8(%rbp)
	jmp	.L387
.L561:
	movq	$256, -8(%rbp)
	jmp	.L387
.L363:
	cmpl	$0, -116(%rbp)
	jns	.L563
	movq	$76, -8(%rbp)
	jmp	.L387
.L563:
	movq	$262, -8(%rbp)
	jmp	.L387
.L344:
	call	rand_b
	movl	%eax, -48(%rbp)
	movq	$31, -8(%rbp)
	jmp	.L387
.L341:
	call	rand_b
	movl	%eax, -16(%rbp)
	movq	$391, -8(%rbp)
	jmp	.L387
.L63:
	cmpl	$0, -116(%rbp)
	jns	.L565
	movq	$378, -8(%rbp)
	jmp	.L387
.L565:
	movq	$414, -8(%rbp)
	jmp	.L387
.L212:
	cmpl	$2, -120(%rbp)
	je	.L567
	movq	$411, -8(%rbp)
	jmp	.L387
.L567:
	movq	$21, -8(%rbp)
	jmp	.L387
.L92:
	movl	-116(%rbp), %eax
	addl	$2, %eax
	cmpl	%eax, -112(%rbp)
	jle	.L569
	movq	$193, -8(%rbp)
	jmp	.L387
.L569:
	movq	$207, -8(%rbp)
	jmp	.L387
.L169:
	addl	$2, -116(%rbp)
	movq	$371, -8(%rbp)
	jmp	.L387
.L12:
	cmpl	$0, -116(%rbp)
	js	.L571
	movq	$333, -8(%rbp)
	jmp	.L387
.L571:
	movq	$351, -8(%rbp)
	jmp	.L387
.L360:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L573
	movq	$373, -8(%rbp)
	jmp	.L387
.L573:
	movq	$310, -8(%rbp)
	jmp	.L387
.L227:
	call	deadlock
	movq	$335, -8(%rbp)
	jmp	.L387
.L135:
	call	deadlock
	movq	$247, -8(%rbp)
	jmp	.L387
.L298:
	call	deadlock
	movq	$408, -8(%rbp)
	jmp	.L387
.L207:
	call	checking_error
	movq	$270, -8(%rbp)
	jmp	.L387
.L373:
	cmpl	$1, -120(%rbp)
	jne	.L575
	movq	$259, -8(%rbp)
	jmp	.L387
.L575:
	movq	$7, -8(%rbp)
	jmp	.L387
.L293:
	cmpl	$0, -116(%rbp)
	jns	.L577
	movq	$221, -8(%rbp)
	jmp	.L387
.L577:
	movq	$197, -8(%rbp)
	jmp	.L387
.L153:
	call	checking_error
	movq	$125, -8(%rbp)
	jmp	.L387
.L142:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	js	.L579
	movq	$75, -8(%rbp)
	jmp	.L387
.L579:
	movq	$17, -8(%rbp)
	jmp	.L387
.L272:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$229, -8(%rbp)
	jmp	.L387
.L312:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$83, -8(%rbp)
	jmp	.L387
.L130:
	movl	-116(%rbp), %eax
	addl	$2, %eax
	cmpl	%eax, -112(%rbp)
	jg	.L581
	movq	$250, -8(%rbp)
	jmp	.L387
.L581:
	movq	$169, -8(%rbp)
	jmp	.L387
.L338:
	call	checking_error
	movq	$210, -8(%rbp)
	jmp	.L387
.L319:
	call	deadlock
	movq	$441, -8(%rbp)
	jmp	.L387
.L198:
	cmpl	$0, -116(%rbp)
	jns	.L583
	movq	$290, -8(%rbp)
	jmp	.L387
.L583:
	movq	$218, -8(%rbp)
	jmp	.L387
.L149:
	call	deadlock
	movq	$256, -8(%rbp)
	jmp	.L387
.L23:
	call	deadlock
	movq	$256, -8(%rbp)
	jmp	.L387
.L5:
	cmpl	$1, -120(%rbp)
	jne	.L585
	movq	$220, -8(%rbp)
	jmp	.L387
.L585:
	movq	$43, -8(%rbp)
	jmp	.L387
.L244:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L587
	movq	$150, -8(%rbp)
	jmp	.L387
.L587:
	movq	$66, -8(%rbp)
	jmp	.L387
.L86:
	call	rand_b
	movl	%eax, -92(%rbp)
	movq	$300, -8(%rbp)
	jmp	.L387
.L359:
	cmpl	$0, -48(%rbp)
	je	.L589
	movq	$134, -8(%rbp)
	jmp	.L387
.L589:
	movq	$129, -8(%rbp)
	jmp	.L387
.L375:
	call	deadlock
	movq	$405, -8(%rbp)
	jmp	.L387
.L204:
	cmpl	$2, -120(%rbp)
	je	.L591
	movq	$246, -8(%rbp)
	jmp	.L387
.L591:
	movq	$390, -8(%rbp)
	jmp	.L387
.L326:
	call	deadlock
	movq	$100, -8(%rbp)
	jmp	.L387
.L211:
	cmpl	$0, -116(%rbp)
	jns	.L593
	movq	$254, -8(%rbp)
	jmp	.L387
.L593:
	movq	$253, -8(%rbp)
	jmp	.L387
.L209:
	call	rand_b
	movl	%eax, -24(%rbp)
	movq	$365, -8(%rbp)
	jmp	.L387
.L306:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$406, -8(%rbp)
	jmp	.L387
.L132:
	cmpl	$0, -116(%rbp)
	jns	.L595
	movq	$175, -8(%rbp)
	jmp	.L387
.L595:
	movq	$49, -8(%rbp)
	jmp	.L387
.L347:
	movl	$2, -120(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L387
.L287:
	call	deadlock
	movq	$397, -8(%rbp)
	jmp	.L387
.L118:
	cmpl	$1, -120(%rbp)
	jne	.L597
	movq	$103, -8(%rbp)
	jmp	.L387
.L597:
	movq	$419, -8(%rbp)
	jmp	.L387
.L277:
	call	rand_b
	movl	%eax, -20(%rbp)
	movq	$423, -8(%rbp)
	jmp	.L387
.L181:
	call	checking_error
	movq	$256, -8(%rbp)
	jmp	.L387
.L74:
	cmpl	$0, -120(%rbp)
	je	.L599
	movq	$195, -8(%rbp)
	jmp	.L387
.L599:
	movq	$145, -8(%rbp)
	jmp	.L387
.L134:
	movl	$2, -120(%rbp)
	movq	$167, -8(%rbp)
	jmp	.L387
.L33:
	call	deadlock
	movq	$21, -8(%rbp)
	jmp	.L387
.L146:
	call	rand_b
	movl	%eax, -64(%rbp)
	movq	$430, -8(%rbp)
	jmp	.L387
.L325:
	movl	$2, -120(%rbp)
	movq	$356, -8(%rbp)
	jmp	.L387
.L271:
	cmpl	$2, -120(%rbp)
	je	.L601
	movq	$79, -8(%rbp)
	jmp	.L387
.L601:
	movq	$441, -8(%rbp)
	jmp	.L387
.L44:
	call	deadlock
	movq	$322, -8(%rbp)
	jmp	.L387
.L364:
	call	checking_error
	movq	$421, -8(%rbp)
	jmp	.L387
.L79:
	call	deadlock
	movq	$193, -8(%rbp)
	jmp	.L387
.L308:
	call	checking_error
	movq	$273, -8(%rbp)
	jmp	.L387
.L295:
	call	deadlock
	movq	$321, -8(%rbp)
	jmp	.L387
.L222:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jle	.L603
	movq	$113, -8(%rbp)
	jmp	.L387
.L603:
	movq	$196, -8(%rbp)
	jmp	.L387
.L241:
	cmpl	$0, -120(%rbp)
	je	.L605
	movq	$297, -8(%rbp)
	jmp	.L387
.L605:
	movq	$357, -8(%rbp)
	jmp	.L387
.L47:
	call	deadlock
	movq	$67, -8(%rbp)
	jmp	.L387
.L37:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	js	.L607
	movq	$174, -8(%rbp)
	jmp	.L387
.L607:
	movq	$275, -8(%rbp)
	jmp	.L387
.L321:
	call	checking_error
	movq	$262, -8(%rbp)
	jmp	.L387
.L215:
	call	rand_b
	movl	%eax, -76(%rbp)
	movq	$424, -8(%rbp)
	jmp	.L387
.L173:
	cmpl	$0, -120(%rbp)
	je	.L609
	movq	$407, -8(%rbp)
	jmp	.L387
.L609:
	movq	$225, -8(%rbp)
	jmp	.L387
.L337:
	cmpl	$0, -104(%rbp)
	je	.L611
	movq	$265, -8(%rbp)
	jmp	.L387
.L611:
	movq	$64, -8(%rbp)
	jmp	.L387
.L252:
	call	checking_error
	movq	$197, -8(%rbp)
	jmp	.L387
.L24:
	call	rand_b
	movl	%eax, -88(%rbp)
	movq	$352, -8(%rbp)
	jmp	.L387
.L162:
	call	rand_b
	movl	%eax, -84(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L387
.L70:
	cmpl	$2, -120(%rbp)
	je	.L613
	movq	$170, -8(%rbp)
	jmp	.L387
.L613:
	movq	$44, -8(%rbp)
	jmp	.L387
.L122:
	cmpl	$2, -120(%rbp)
	je	.L615
	movq	$95, -8(%rbp)
	jmp	.L387
.L615:
	movq	$133, -8(%rbp)
	jmp	.L387
.L304:
	cmpl	$0, -116(%rbp)
	jns	.L617
	movq	$252, -8(%rbp)
	jmp	.L387
.L617:
	movq	$273, -8(%rbp)
	jmp	.L387
.L199:
	call	deadlock
	movq	$61, -8(%rbp)
	jmp	.L387
.L210:
	cmpl	$0, -116(%rbp)
	jns	.L619
	movq	$154, -8(%rbp)
	jmp	.L387
.L619:
	movq	$256, -8(%rbp)
	jmp	.L387
.L377:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L621
	movq	$226, -8(%rbp)
	jmp	.L387
.L621:
	movq	$36, -8(%rbp)
	jmp	.L387
.L25:
	call	checking_error
	movq	$341, -8(%rbp)
	jmp	.L387
.L374:
	call	checking_error
	movq	$341, -8(%rbp)
	jmp	.L387
.L196:
	cmpl	$0, -116(%rbp)
	js	.L623
	movq	$291, -8(%rbp)
	jmp	.L387
.L623:
	movq	$120, -8(%rbp)
	jmp	.L387
.L83:
	cmpl	$0, -28(%rbp)
	je	.L625
	movq	$367, -8(%rbp)
	jmp	.L387
.L625:
	movq	$188, -8(%rbp)
	jmp	.L387
.L342:
	call	checking_error
	movq	$121, -8(%rbp)
	jmp	.L387
.L297:
	cmpl	$0, -116(%rbp)
	jns	.L627
	movq	$353, -8(%rbp)
	jmp	.L387
.L627:
	movq	$422, -8(%rbp)
	jmp	.L387
.L81:
	movl	$2, -120(%rbp)
	movq	$306, -8(%rbp)
	jmp	.L387
.L59:
	cmpl	$1, -120(%rbp)
	jne	.L629
	movq	$189, -8(%rbp)
	jmp	.L387
.L629:
	movq	$251, -8(%rbp)
	jmp	.L387
.L260:
	call	deadlock
	movq	$150, -8(%rbp)
	jmp	.L387
.L50:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$87, -8(%rbp)
	jmp	.L387
.L106:
	cmpl	$0, -80(%rbp)
	je	.L631
	movq	$287, -8(%rbp)
	jmp	.L387
.L631:
	movq	$443, -8(%rbp)
	jmp	.L387
.L117:
	cmpl	$0, -116(%rbp)
	jns	.L633
	movq	$274, -8(%rbp)
	jmp	.L387
.L633:
	movq	$309, -8(%rbp)
	jmp	.L387
.L311:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L635
	movq	$243, -8(%rbp)
	jmp	.L387
.L635:
	movq	$33, -8(%rbp)
	jmp	.L387
.L102:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$339, -8(%rbp)
	jmp	.L387
.L94:
	call	deadlock
	movq	$386, -8(%rbp)
	jmp	.L387
.L151:
	call	checking_error
	movq	$210, -8(%rbp)
	jmp	.L387
.L51:
	cmpl	$0, -116(%rbp)
	jns	.L637
	movq	$340, -8(%rbp)
	jmp	.L387
.L637:
	movq	$414, -8(%rbp)
	jmp	.L387
.L203:
	call	rand_b
	movl	%eax, -68(%rbp)
	movq	$139, -8(%rbp)
	jmp	.L387
.L328:
	addl	$2, -116(%rbp)
	movq	$202, -8(%rbp)
	jmp	.L387
.L273:
	call	deadlock
	movq	$313, -8(%rbp)
	jmp	.L387
.L294:
	call	deadlock
	movq	$224, -8(%rbp)
	jmp	.L387
.L219:
	addl	$2, -116(%rbp)
	movq	$157, -8(%rbp)
	jmp	.L387
.L88:
	cmpl	$0, -116(%rbp)
	jns	.L639
	movq	$56, -8(%rbp)
	jmp	.L387
.L639:
	movq	$210, -8(%rbp)
	jmp	.L387
.L82:
	cmpl	$0, -12(%rbp)
	je	.L641
	movq	$52, -8(%rbp)
	jmp	.L387
.L641:
	movq	$122, -8(%rbp)
	jmp	.L387
.L335:
	cmpl	$2, -120(%rbp)
	je	.L643
	movq	$118, -8(%rbp)
	jmp	.L387
.L643:
	movq	$397, -8(%rbp)
	jmp	.L387
.L243:
	call	deadlock
	movq	$61, -8(%rbp)
	jmp	.L387
.L103:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$323, -8(%rbp)
	jmp	.L387
.L216:
	call	deadlock
	movq	$113, -8(%rbp)
	jmp	.L387
.L193:
	call	rand_b
	movl	%eax, -56(%rbp)
	movq	$234, -8(%rbp)
	jmp	.L387
.L108:
	call	rand_b
	movl	%eax, -108(%rbp)
	movq	$50, -8(%rbp)
	jmp	.L387
.L42:
	call	deadlock
	movq	$343, -8(%rbp)
	jmp	.L387
.L380:
	call	deadlock
	movq	$100, -8(%rbp)
	jmp	.L387
.L253:
	cmpl	$2, -120(%rbp)
	je	.L645
	movq	$368, -8(%rbp)
	jmp	.L387
.L645:
	movq	$160, -8(%rbp)
	jmp	.L387
.L113:
	call	checking_error
	movq	$125, -8(%rbp)
	jmp	.L387
.L55:
	cmpl	$0, -116(%rbp)
	jns	.L647
	movq	$47, -8(%rbp)
	jmp	.L387
.L647:
	movq	$257, -8(%rbp)
	jmp	.L387
.L34:
	call	checking_error
	movq	$121, -8(%rbp)
	jmp	.L387
.L10:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L649
	movq	$232, -8(%rbp)
	jmp	.L387
.L649:
	movq	$54, -8(%rbp)
	jmp	.L387
.L289:
	call	deadlock
	movq	$260, -8(%rbp)
	jmp	.L387
.L121:
	cmpl	$0, -92(%rbp)
	je	.L651
	movq	$255, -8(%rbp)
	jmp	.L387
.L651:
	movq	$362, -8(%rbp)
	jmp	.L387
.L254:
	call	checking_error
	movq	$256, -8(%rbp)
	jmp	.L387
.L256:
	cmpl	$0, -120(%rbp)
	je	.L653
	movq	$435, -8(%rbp)
	jmp	.L387
.L653:
	movq	$132, -8(%rbp)
	jmp	.L387
.L85:
	call	deadlock
	movq	$315, -8(%rbp)
	jmp	.L387
.L54:
	call	checking_error
	movq	$218, -8(%rbp)
	jmp	.L387
.L38:
	cmpl	$0, -116(%rbp)
	jns	.L655
	movq	$20, -8(%rbp)
	jmp	.L387
.L655:
	movq	$421, -8(%rbp)
	jmp	.L387
.L353:
	cmpl	$0, -116(%rbp)
	jns	.L657
	movq	$413, -8(%rbp)
	jmp	.L387
.L657:
	movq	$53, -8(%rbp)
	jmp	.L387
.L333:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$432, -8(%rbp)
	jmp	.L387
.L250:
	cmpl	$2, -120(%rbp)
	je	.L659
	movq	$230, -8(%rbp)
	jmp	.L387
.L659:
	movq	$96, -8(%rbp)
	jmp	.L387
.L291:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$72, -8(%rbp)
	jmp	.L387
.L268:
	addl	$1, -116(%rbp)
	addl	$1, -112(%rbp)
	movq	$212, -8(%rbp)
	jmp	.L387
.L163:
	call	rand_b
	movl	%eax, -12(%rbp)
	movq	$345, -8(%rbp)
	jmp	.L387
.L60:
	call	deadlock
	movq	$61, -8(%rbp)
	jmp	.L387
.L141:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L661
	movq	$162, -8(%rbp)
	jmp	.L387
.L661:
	movq	$85, -8(%rbp)
	jmp	.L387
.L336:
	cmpl	$0, -116(%rbp)
	js	.L663
	movq	$144, -8(%rbp)
	jmp	.L387
.L663:
	movq	$222, -8(%rbp)
	jmp	.L387
.L238:
	call	deadlock
	movq	$44, -8(%rbp)
	jmp	.L387
.L292:
	cmpl	$1, -120(%rbp)
	jne	.L665
	movq	$329, -8(%rbp)
	jmp	.L387
.L665:
	movq	$198, -8(%rbp)
	jmp	.L387
.L231:
	call	checking_error
	movq	$256, -8(%rbp)
	jmp	.L387
.L164:
	cmpl	$1, -120(%rbp)
	jne	.L667
	movq	$117, -8(%rbp)
	jmp	.L387
.L667:
	movq	$284, -8(%rbp)
	jmp	.L387
.L322:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jle	.L669
	movq	$386, -8(%rbp)
	jmp	.L387
.L669:
	movq	$331, -8(%rbp)
	jmp	.L387
.L282:
	movq	$358, -8(%rbp)
	jmp	.L387
.L152:
	call	deadlock
	movq	$37, -8(%rbp)
	jmp	.L387
.L345:
	cmpl	$0, -36(%rbp)
	je	.L671
	movq	$65, -8(%rbp)
	jmp	.L387
.L671:
	movq	$8, -8(%rbp)
	jmp	.L387
.L69:
	cmpl	$1, -120(%rbp)
	jne	.L673
	movq	$248, -8(%rbp)
	jmp	.L387
.L673:
	movq	$73, -8(%rbp)
	jmp	.L387
.L267:
	cmpl	$0, -68(%rbp)
	je	.L675
	movq	$429, -8(%rbp)
	jmp	.L387
.L675:
	movq	$41, -8(%rbp)
	jmp	.L387
.L259:
	cmpl	$0, -120(%rbp)
	je	.L677
	movq	$86, -8(%rbp)
	jmp	.L387
.L677:
	movq	$318, -8(%rbp)
	jmp	.L387
.L182:
	cmpl	$0, -56(%rbp)
	je	.L679
	movq	$209, -8(%rbp)
	jmp	.L387
.L679:
	movq	$418, -8(%rbp)
	jmp	.L387
.L128:
	call	checking_error
	movq	$314, -8(%rbp)
	jmp	.L387
.L6:
	call	deadlock
	movq	$100, -8(%rbp)
	jmp	.L387
.L159:
	movl	-112(%rbp), %eax
	subl	%eax, -116(%rbp)
	movl	$1, -120(%rbp)
	movq	$264, -8(%rbp)
	jmp	.L387
.L340:
	call	rand_b
	movl	%eax, -96(%rbp)
	movq	$324, -8(%rbp)
	jmp	.L387
.L57:
	cmpl	$0, -116(%rbp)
	jns	.L681
	movq	$181, -8(%rbp)
	jmp	.L387
.L681:
	movq	$49, -8(%rbp)
	jmp	.L387
.L221:
	call	deadlock
	movq	$113, -8(%rbp)
	jmp	.L387
.L197:
	call	rand_b
	movl	%eax, -100(%rbp)
	movq	$409, -8(%rbp)
	jmp	.L387
.L110:
	cmpl	$0, -116(%rbp)
	jns	.L683
	movq	$15, -8(%rbp)
	jmp	.L387
.L683:
	movq	$439, -8(%rbp)
	jmp	.L387
.L346:
	call	checking_error
	movq	$257, -8(%rbp)
	jmp	.L387
.L323:
	call	deadlock
	movq	$45, -8(%rbp)
	jmp	.L387
.L61:
	movl	$2, -120(%rbp)
	movq	$327, -8(%rbp)
	jmp	.L387
.L310:
	call	checking_error
	movq	$256, -8(%rbp)
	jmp	.L387
.L137:
	call	deadlock
	movq	$67, -8(%rbp)
	jmp	.L387
.L36:
	addl	$2, -116(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L387
.L305:
	cmpl	$0, -116(%rbp)
	jns	.L685
	movq	$392, -8(%rbp)
	jmp	.L387
.L685:
	movq	$257, -8(%rbp)
	jmp	.L387
.L245:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L687
	movq	$37, -8(%rbp)
	jmp	.L387
.L687:
	movq	$171, -8(%rbp)
	jmp	.L387
.L186:
	call	deadlock
	movq	$96, -8(%rbp)
	jmp	.L387
.L303:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -112(%rbp)
	jg	.L689
	movq	$315, -8(%rbp)
	jmp	.L387
.L689:
	movq	$342, -8(%rbp)
	jmp	.L387
.L290:
	call	rand_b
	movl	%eax, -104(%rbp)
	movq	$57, -8(%rbp)
	jmp	.L387
.L354:
	movl	$2, -120(%rbp)
	movq	$111, -8(%rbp)
	jmp	.L387
.L64:
	call	deadlock
	movq	$160, -8(%rbp)
	jmp	.L387
.L167:
	call	checking_error
	movq	$273, -8(%rbp)
	jmp	.L387
.L154:
	cmpl	$2, -120(%rbp)
	je	.L691
	movq	$12, -8(%rbp)
	jmp	.L387
.L691:
	movq	$405, -8(%rbp)
	jmp	.L387
.L170:
	movl	-116(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jl	.L693
	movq	$45, -8(%rbp)
	jmp	.L387
.L693:
	movq	$211, -8(%rbp)
	jmp	.L387
.L98:
	cmpl	$0, -116(%rbp)
	jns	.L695
	movq	$131, -8(%rbp)
	jmp	.L387
.L695:
	movq	$223, -8(%rbp)
	jmp	.L387
.L351:
	cmpl	$1, -120(%rbp)
	jne	.L697
	movq	$444, -8(%rbp)
	jmp	.L387
.L697:
	movq	$239, -8(%rbp)
	jmp	.L387
.L136:
	call	checking_error
	movq	$256, -8(%rbp)
	jmp	.L387
.L307:
	call	deadlock
	movq	$133, -8(%rbp)
	jmp	.L387
.L258:
	call	deadlock
	movq	$427, -8(%rbp)
	jmp	.L387
.L214:
	call	deadlock
	movq	$408, -8(%rbp)
	jmp	.L387
.L184:
	movl	$2, -120(%rbp)
	movq	$311, -8(%rbp)
	jmp	.L387
.L131:
	call	checking_error
	movq	$218, -8(%rbp)
	jmp	.L387
.L350:
	call	deadlock
	movq	$271, -8(%rbp)
	jmp	.L387
.L166:
	call	rand_b
	movl	%eax, -52(%rbp)
	movq	$281, -8(%rbp)
	jmp	.L387
.L133:
	cmpl	$1, -120(%rbp)
	jne	.L699
	movq	$168, -8(%rbp)
	jmp	.L387
.L699:
	movq	$6, -8(%rbp)
	jmp	.L387
.L100:
	cmpl	$0, -96(%rbp)
	je	.L701
	movq	$124, -8(%rbp)
	jmp	.L387
.L701:
	movq	$70, -8(%rbp)
	jmp	.L387
.L187:
	cmpl	$0, -116(%rbp)
	jns	.L703
	movq	$384, -8(%rbp)
	jmp	.L387
.L703:
	movq	$218, -8(%rbp)
	jmp	.L387
.L232:
	call	rand_b
	movl	%eax, -28(%rbp)
	movq	$344, -8(%rbp)
	jmp	.L387
.L329:
	call	deadlock
	movq	$150, -8(%rbp)
	jmp	.L387
.L317:
	cmpl	$0, -116(%rbp)
	jns	.L705
	movq	$236, -8(%rbp)
	jmp	.L387
.L705:
	movq	$256, -8(%rbp)
	jmp	.L387
.L68:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	jns	.L707
	movq	$164, -8(%rbp)
	jmp	.L387
.L707:
	movq	$146, -8(%rbp)
	jmp	.L387
.L87:
	call	checking_error
	movq	$414, -8(%rbp)
	jmp	.L387
.L356:
	cmpl	$0, -120(%rbp)
	je	.L709
	movq	$233, -8(%rbp)
	jmp	.L387
.L709:
	movq	$332, -8(%rbp)
	jmp	.L387
.L156:
	call	deadlock
	movq	$322, -8(%rbp)
	jmp	.L387
.L179:
	call	deadlock
	movq	$89, -8(%rbp)
	jmp	.L387
.L349:
	call	deadlock
	movq	$250, -8(%rbp)
	jmp	.L387
.L314:
	movl	-116(%rbp), %eax
	subl	-112(%rbp), %eax
	testl	%eax, %eax
	js	.L711
	movq	$142, -8(%rbp)
	jmp	.L387
.L711:
	movq	$401, -8(%rbp)
	jmp	.L387
.L384:
	call	deadlock
	movq	$138, -8(%rbp)
	jmp	.L387
.L714:
	nop
.L387:
	jmp	.L713
.L715:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	ts_restructured, .-ts_restructured
	.section	.rodata
.LC0:
	.string	"checking error"
	.text
	.globl	checking_error
	.type	checking_error, @function
checking_error:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L721:
	cmpq	$0, -8(%rbp)
	je	.L717
	cmpq	$2, -8(%rbp)
	je	.L718
	jmp	.L720
.L717:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$14, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$2, %edi
	call	exit@PLT
.L718:
	movq	$0, -8(%rbp)
	nop
.L720:
	jmp	.L721
	.cfi_endproc
.LFE3:
	.size	checking_error, .-checking_error
	.globl	rand_z
	.type	rand_z, @function
rand_z:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L728:
	cmpq	$2, -8(%rbp)
	je	.L723
	cmpq	$2, -8(%rbp)
	ja	.L730
	cmpq	$0, -8(%rbp)
	je	.L725
	cmpq	$1, -8(%rbp)
	jne	.L730
	movq	$0, -8(%rbp)
	jmp	.L726
.L725:
	call	rand@PLT
	movl	%eax, -16(%rbp)
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L726
.L723:
	movl	-16(%rbp), %eax
	subl	-12(%rbp), %eax
	jmp	.L729
.L730:
	nop
.L726:
	jmp	.L728
.L729:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	rand_z, .-rand_z
	.section	.rodata
.LC1:
	.string	"deadlock"
	.text
	.globl	deadlock
	.type	deadlock, @function
deadlock:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L735:
	cmpq	$2, -8(%rbp)
	je	.L732
	cmpq	$3, -8(%rbp)
	jne	.L736
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L734
.L732:
	movq	$2, -8(%rbp)
	jmp	.L734
.L736:
	nop
.L734:
	jmp	.L735
	.cfi_endproc
.LFE9:
	.size	deadlock, .-deadlock
	.globl	main
	.type	main, @function
main:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_fpUC_envp(%rip)
	nop
.L738:
	movq	$0, _TIG_IZ_fpUC_argv(%rip)
	nop
.L739:
	movl	$0, _TIG_IZ_fpUC_argc(%rip)
	nop
	nop
.L740:
.L741:
#APP
# 83 "Par4All_par4all_bardin_thesis_2-03.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-fpUC--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_fpUC_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_fpUC_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_fpUC_envp(%rip)
	nop
	movq	$2, -8(%rbp)
.L747:
	cmpq	$2, -8(%rbp)
	je	.L742
	cmpq	$2, -8(%rbp)
	ja	.L749
	cmpq	$0, -8(%rbp)
	je	.L744
	cmpq	$1, -8(%rbp)
	jne	.L749
	movl	$0, %eax
	jmp	.L748
.L744:
	call	ts_singlestate
	call	ts_restructured
	movq	$1, -8(%rbp)
	jmp	.L746
.L742:
	movq	$0, -8(%rbp)
	jmp	.L746
.L749:
	nop
.L746:
	jmp	.L747
.L748:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	main, .-main
	.globl	ts_singlestate
	.type	ts_singlestate, @function
ts_singlestate:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	$23, -8(%rbp)
.L823:
	cmpq	$41, -8(%rbp)
	ja	.L824
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L753(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L753(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L753:
	.long	.L790-.L753
	.long	.L789-.L753
	.long	.L788-.L753
	.long	.L787-.L753
	.long	.L824-.L753
	.long	.L786-.L753
	.long	.L785-.L753
	.long	.L824-.L753
	.long	.L784-.L753
	.long	.L783-.L753
	.long	.L782-.L753
	.long	.L781-.L753
	.long	.L780-.L753
	.long	.L779-.L753
	.long	.L778-.L753
	.long	.L777-.L753
	.long	.L776-.L753
	.long	.L775-.L753
	.long	.L774-.L753
	.long	.L773-.L753
	.long	.L772-.L753
	.long	.L771-.L753
	.long	.L770-.L753
	.long	.L769-.L753
	.long	.L768-.L753
	.long	.L767-.L753
	.long	.L766-.L753
	.long	.L765-.L753
	.long	.L824-.L753
	.long	.L764-.L753
	.long	.L763-.L753
	.long	.L762-.L753
	.long	.L761-.L753
	.long	.L825-.L753
	.long	.L759-.L753
	.long	.L824-.L753
	.long	.L758-.L753
	.long	.L757-.L753
	.long	.L756-.L753
	.long	.L755-.L753
	.long	.L754-.L753
	.long	.L752-.L753
	.text
.L774:
	call	deadlock
	movq	$39, -8(%rbp)
	jmp	.L791
.L767:
	movl	$1, -36(%rbp)
	call	rand@PLT
	movl	%eax, -32(%rbp)
	call	rand@PLT
	movl	%eax, -28(%rbp)
	movq	$37, -8(%rbp)
	jmp	.L791
.L763:
	cmpl	$0, -32(%rbp)
	jns	.L792
	movq	$16, -8(%rbp)
	jmp	.L791
.L792:
	movq	$17, -8(%rbp)
	jmp	.L791
.L778:
	cmpl	$0, -12(%rbp)
	je	.L794
	movq	$11, -8(%rbp)
	jmp	.L791
.L794:
	movq	$33, -8(%rbp)
	jmp	.L791
.L777:
	call	deadlock
	movq	$0, -8(%rbp)
	jmp	.L791
.L762:
	cmpl	$2, -36(%rbp)
	je	.L796
	movq	$26, -8(%rbp)
	jmp	.L791
.L796:
	movq	$10, -8(%rbp)
	jmp	.L791
.L780:
	call	checking_error
	movq	$17, -8(%rbp)
	jmp	.L791
.L784:
	call	rand_b
	movl	%eax, -24(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L791
.L789:
	call	checking_error
	movq	$17, -8(%rbp)
	jmp	.L791
.L769:
	movq	$25, -8(%rbp)
	jmp	.L791
.L787:
	cmpl	$0, -32(%rbp)
	jns	.L798
	movq	$24, -8(%rbp)
	jmp	.L791
.L798:
	movq	$17, -8(%rbp)
	jmp	.L791
.L776:
	call	checking_error
	movq	$17, -8(%rbp)
	jmp	.L791
.L768:
	call	checking_error
	movq	$17, -8(%rbp)
	jmp	.L791
.L771:
	call	deadlock
	movq	$0, -8(%rbp)
	jmp	.L791
.L758:
	movl	$2, -36(%rbp)
	movq	$40, -8(%rbp)
	jmp	.L791
.L766:
	call	deadlock
	movq	$10, -8(%rbp)
	jmp	.L791
.L781:
	call	rand_b
	movl	%eax, -16(%rbp)
	movq	$38, -8(%rbp)
	jmp	.L791
.L783:
	cmpl	$0, -32(%rbp)
	js	.L800
	movq	$0, -8(%rbp)
	jmp	.L791
.L800:
	movq	$21, -8(%rbp)
	jmp	.L791
.L779:
	call	rand_b
	movl	%eax, -20(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L791
.L773:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jl	.L802
	movq	$39, -8(%rbp)
	jmp	.L791
.L802:
	movq	$22, -8(%rbp)
	jmp	.L791
.L761:
	cmpl	$0, -32(%rbp)
	jns	.L804
	movq	$6, -8(%rbp)
	jmp	.L791
.L804:
	movq	$17, -8(%rbp)
	jmp	.L791
.L775:
	call	rand_b
	movl	%eax, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L791
.L754:
	cmpl	$0, -36(%rbp)
	je	.L806
	movq	$41, -8(%rbp)
	jmp	.L791
.L806:
	movq	$2, -8(%rbp)
	jmp	.L791
.L785:
	call	checking_error
	movq	$17, -8(%rbp)
	jmp	.L791
.L765:
	cmpl	$0, -20(%rbp)
	je	.L808
	movq	$31, -8(%rbp)
	jmp	.L791
.L808:
	movq	$8, -8(%rbp)
	jmp	.L791
.L756:
	cmpl	$0, -16(%rbp)
	je	.L810
	movq	$34, -8(%rbp)
	jmp	.L791
.L810:
	movq	$13, -8(%rbp)
	jmp	.L791
.L759:
	cmpl	$1, -36(%rbp)
	jne	.L812
	movq	$9, -8(%rbp)
	jmp	.L791
.L812:
	movq	$15, -8(%rbp)
	jmp	.L791
.L770:
	call	deadlock
	movq	$39, -8(%rbp)
	jmp	.L791
.L786:
	cmpl	$0, -24(%rbp)
	je	.L814
	movq	$29, -8(%rbp)
	jmp	.L791
.L814:
	movq	$36, -8(%rbp)
	jmp	.L791
.L757:
	cmpl	$0, -32(%rbp)
	jns	.L817
	movq	$1, -8(%rbp)
	jmp	.L791
.L817:
	movq	$17, -8(%rbp)
	jmp	.L791
.L752:
	movl	-28(%rbp), %eax
	subl	%eax, -32(%rbp)
	movl	$1, -36(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L791
.L782:
	addl	$1, -32(%rbp)
	addl	$1, -28(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L791
.L790:
	addl	$2, -32(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L791
.L755:
	movl	$2, -36(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L791
.L764:
	cmpl	$1, -36(%rbp)
	jne	.L819
	movq	$19, -8(%rbp)
	jmp	.L791
.L819:
	movq	$18, -8(%rbp)
	jmp	.L791
.L788:
	call	deadlock
	movq	$41, -8(%rbp)
	jmp	.L791
.L772:
	cmpl	$0, -32(%rbp)
	jns	.L821
	movq	$12, -8(%rbp)
	jmp	.L791
.L821:
	movq	$17, -8(%rbp)
	jmp	.L791
.L824:
	nop
.L791:
	jmp	.L823
.L825:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	ts_singlestate, .-ts_singlestate
	.globl	rand_b
	.type	rand_b, @function
rand_b:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L831:
	cmpq	$0, -8(%rbp)
	je	.L827
	cmpq	$1, -8(%rbp)
	jne	.L833
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L829
.L827:
	movl	-12(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	jmp	.L832
.L833:
	nop
.L829:
	jmp	.L831
.L832:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	rand_b, .-rand_b
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
