	.file	"abhaysaxenaa_Word-Wrapping_ww_flatten.c"
	.text
	.globl	_TIG_IZ_d9JB_argc
	.bss
	.align 4
	.type	_TIG_IZ_d9JB_argc, @object
	.size	_TIG_IZ_d9JB_argc, 4
_TIG_IZ_d9JB_argc:
	.zero	4
	.globl	filename
	.align 8
	.type	filename, @object
	.size	filename, 8
filename:
	.zero	8
	.globl	_TIG_IZ_d9JB_argv
	.align 8
	.type	_TIG_IZ_d9JB_argv, @object
	.size	_TIG_IZ_d9JB_argv, 8
_TIG_IZ_d9JB_argv:
	.zero	8
	.globl	exit_status
	.align 4
	.type	exit_status, @object
	.size	exit_status, 4
exit_status:
	.zero	4
	.globl	first_text_found
	.align 4
	.type	first_text_found, @object
	.size	first_text_found, 4
first_text_found:
	.zero	4
	.globl	open_file
	.align 4
	.type	open_file, @object
	.size	open_file, 4
open_file:
	.zero	4
	.globl	data
	.align 32
	.type	data, @object
	.size	data, 144
data:
	.zero	144
	.globl	err
	.align 4
	.type	err, @object
	.size	err, 4
err:
	.zero	4
	.globl	space_ct
	.align 4
	.type	space_ct, @object
	.size	space_ct, 4
space_ct:
	.zero	4
	.globl	close_status
	.align 4
	.type	close_status, @object
	.size	close_status, 4
close_status:
	.zero	4
	.globl	nrd
	.align 8
	.type	nrd, @object
	.size	nrd, 8
nrd:
	.zero	8
	.globl	_TIG_IZ_d9JB_envp
	.align 8
	.type	_TIG_IZ_d9JB_envp, @object
	.size	_TIG_IZ_d9JB_envp, 8
_TIG_IZ_d9JB_envp:
	.zero	8
	.globl	accumulator
	.align 4
	.type	accumulator, @object
	.size	accumulator, 4
accumulator:
	.zero	4
	.globl	path
	.align 8
	.type	path, @object
	.size	path, 8
path:
	.zero	8
	.globl	bytes_read
	.align 4
	.type	bytes_read, @object
	.size	bytes_read, 4
bytes_read:
	.zero	4
	.globl	sb
	.align 16
	.type	sb, @object
	.size	sb, 24
sb:
	.zero	24
	.globl	buffer
	.align 8
	.type	buffer, @object
	.size	buffer, 8
buffer:
	.zero	8
	.globl	newline_ct
	.align 4
	.type	newline_ct, @object
	.size	newline_ct, 4
newline_ct:
	.zero	4
	.text
	.globl	strbuf_append
	.type	strbuf_append, @function
strbuf_append:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, %eax
	movb	%al, -44(%rbp)
	movq	$1, -16(%rbp)
.L17:
	cmpq	$9, -16(%rbp)
	ja	.L18
	movq	-16(%rbp), %rax
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
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L18-.L4
	.long	.L18-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L18-.L4
	.long	.L3-.L4
	.text
.L8:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	addq	%rax, %rax
	movq	%rax, -32(%rbp)
	movq	-40(%rbp), %rax
	movq	16(%rax), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L11
.L9:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	cmpq	%rax, %rdx
	jne	.L12
	movq	$4, -16(%rbp)
	jmp	.L11
.L12:
	movq	$0, -16(%rbp)
	jmp	.L11
.L3:
	movl	$0, %eax
	jmp	.L14
.L6:
	movl	$1, %eax
	jmp	.L14
.L7:
	movq	-40(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-40(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$0, -16(%rbp)
	jmp	.L11
.L10:
	movq	-40(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	addq	%rax, %rdx
	movzbl	-44(%rbp), %eax
	movb	%al, (%rdx)
	movq	-40(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	addq	$1, %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$9, -16(%rbp)
	jmp	.L11
.L5:
	cmpq	$0, -24(%rbp)
	jne	.L15
	movq	$6, -16(%rbp)
	jmp	.L11
.L15:
	movq	$5, -16(%rbp)
	jmp	.L11
.L18:
	nop
.L11:
	jmp	.L17
.L14:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	strbuf_append, .-strbuf_append
	.section	.rodata
	.align 8
.LC0:
	.string	"ERROR: Path file could not be opened"
	.align 8
.LC1:
	.string	"ERROR: File on path could not be closed"
	.align 8
.LC2:
	.string	"ERROR: Directory specified could not be opened."
	.align 8
.LC3:
	.string	"Invalid argument. Enter a valid byte size."
	.align 8
.LC4:
	.string	"ERROR: Cannot open file specified in the path."
	.align 8
.LC5:
	.string	"ERROR: Wrap file on path could not be closed"
.LC6:
	.string	"."
	.align 8
.LC7:
	.string	"ERROR: Cannot close file specified in path..\n"
	.align 8
.LC8:
	.string	"ERROR: Wrap file could not be opened or created"
	.align 8
.LC9:
	.string	"ERROR: Directory specified could not be closed"
.LC10:
	.string	"wrap."
	.align 8
.LC11:
	.string	"ERROR: insufficient number of arguments."
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$224, %rsp
	movl	%edi, -196(%rbp)
	movq	%rsi, -208(%rbp)
	movq	%rdx, -216(%rbp)
	movq	$0, nrd(%rip)
	nop
.L20:
	movl	$0, err(%rip)
	nop
.L21:
	movq	$0, data(%rip)
	movq	$0, 8+data(%rip)
	movq	$0, 16+data(%rip)
	movl	$0, 24+data(%rip)
	movl	$0, 28+data(%rip)
	movl	$0, 32+data(%rip)
	movl	$0, 36+data(%rip)
	movq	$0, 40+data(%rip)
	movq	$0, 48+data(%rip)
	movq	$0, 56+data(%rip)
	movq	$0, 64+data(%rip)
	movq	$0, 72+data(%rip)
	movq	$0, 80+data(%rip)
	movq	$0, 88+data(%rip)
	movq	$0, 96+data(%rip)
	movq	$0, 104+data(%rip)
	movq	$0, 112+data(%rip)
	movq	$0, 120+data(%rip)
	movq	$0, 128+data(%rip)
	movq	$0, 136+data(%rip)
	nop
.L22:
	movq	$0, path(%rip)
	nop
.L23:
	movl	$0, first_text_found(%rip)
	nop
.L24:
	movl	$0, exit_status(%rip)
	nop
.L25:
	movl	$0, close_status(%rip)
	nop
.L26:
	movl	$0, open_file(%rip)
	nop
.L27:
	movl	$0, bytes_read(%rip)
	nop
.L28:
	movl	$0, accumulator(%rip)
	nop
.L29:
	movl	$0, newline_ct(%rip)
	nop
.L30:
	movl	$0, space_ct(%rip)
	nop
.L31:
	movq	$0, buffer(%rip)
	nop
.L32:
	movq	$0, filename(%rip)
	nop
.L33:
	movq	$0, sb(%rip)
	movq	$0, 8+sb(%rip)
	movq	$0, 16+sb(%rip)
	nop
.L34:
	movq	$0, _TIG_IZ_d9JB_envp(%rip)
	nop
.L35:
	movq	$0, _TIG_IZ_d9JB_argv(%rip)
	nop
.L36:
	movl	$0, _TIG_IZ_d9JB_argc(%rip)
	nop
	nop
.L37:
.L38:
#APP
# 325 "abhaysaxenaa_Word-Wrapping_ww.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-d9JB--0
# 0 "" 2
#NO_APP
	movl	-196(%rbp), %eax
	movl	%eax, _TIG_IZ_d9JB_argc(%rip)
	movq	-208(%rbp), %rax
	movq	%rax, _TIG_IZ_d9JB_argv(%rip)
	movq	-216(%rbp), %rax
	movq	%rax, _TIG_IZ_d9JB_envp(%rip)
	nop
	movq	$60, -96(%rbp)
.L167:
	cmpq	$97, -96(%rbp)
	ja	.L168
	movq	-96(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L41(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L41(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L41:
	.long	.L114-.L41
	.long	.L113-.L41
	.long	.L112-.L41
	.long	.L111-.L41
	.long	.L110-.L41
	.long	.L109-.L41
	.long	.L108-.L41
	.long	.L107-.L41
	.long	.L106-.L41
	.long	.L168-.L41
	.long	.L105-.L41
	.long	.L168-.L41
	.long	.L104-.L41
	.long	.L103-.L41
	.long	.L168-.L41
	.long	.L168-.L41
	.long	.L102-.L41
	.long	.L101-.L41
	.long	.L100-.L41
	.long	.L99-.L41
	.long	.L168-.L41
	.long	.L168-.L41
	.long	.L98-.L41
	.long	.L97-.L41
	.long	.L96-.L41
	.long	.L95-.L41
	.long	.L94-.L41
	.long	.L93-.L41
	.long	.L92-.L41
	.long	.L168-.L41
	.long	.L168-.L41
	.long	.L91-.L41
	.long	.L168-.L41
	.long	.L168-.L41
	.long	.L90-.L41
	.long	.L89-.L41
	.long	.L168-.L41
	.long	.L88-.L41
	.long	.L87-.L41
	.long	.L168-.L41
	.long	.L86-.L41
	.long	.L85-.L41
	.long	.L84-.L41
	.long	.L168-.L41
	.long	.L83-.L41
	.long	.L82-.L41
	.long	.L81-.L41
	.long	.L168-.L41
	.long	.L80-.L41
	.long	.L168-.L41
	.long	.L79-.L41
	.long	.L78-.L41
	.long	.L77-.L41
	.long	.L76-.L41
	.long	.L75-.L41
	.long	.L74-.L41
	.long	.L73-.L41
	.long	.L72-.L41
	.long	.L71-.L41
	.long	.L70-.L41
	.long	.L69-.L41
	.long	.L68-.L41
	.long	.L67-.L41
	.long	.L66-.L41
	.long	.L65-.L41
	.long	.L64-.L41
	.long	.L63-.L41
	.long	.L62-.L41
	.long	.L61-.L41
	.long	.L60-.L41
	.long	.L168-.L41
	.long	.L59-.L41
	.long	.L58-.L41
	.long	.L57-.L41
	.long	.L56-.L41
	.long	.L55-.L41
	.long	.L54-.L41
	.long	.L168-.L41
	.long	.L53-.L41
	.long	.L52-.L41
	.long	.L168-.L41
	.long	.L168-.L41
	.long	.L51-.L41
	.long	.L50-.L41
	.long	.L49-.L41
	.long	.L48-.L41
	.long	.L168-.L41
	.long	.L47-.L41
	.long	.L46-.L41
	.long	.L45-.L41
	.long	.L168-.L41
	.long	.L168-.L41
	.long	.L44-.L41
	.long	.L168-.L41
	.long	.L43-.L41
	.long	.L168-.L41
	.long	.L42-.L41
	.long	.L40-.L41
	.text
.L100:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$59, -96(%rbp)
	jmp	.L115
.L79:
	movq	path(%rip), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -64(%rbp)
	movq	-120(%rbp), %rax
	addq	$19, %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -56(%rbp)
	movq	-64(%rbp), %rax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	addl	%edx, %eax
	addl	$7, %eax
	movl	%eax, -144(%rbp)
	movl	-144(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -112(%rbp)
	movq	path(%rip), %rdx
	movq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-112(%rbp), %rax
	addq	%rdx, %rax
	movl	$1634891567, (%rax)
	movw	$11888, 4(%rax)
	movb	$0, 6(%rax)
	movq	-120(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	path(%rip), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	-120(%rbp), %rax
	addq	$19, %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -32(%rbp)
	movq	-40(%rbp), %rax
	movl	%eax, %edx
	movq	-32(%rbp), %rax
	addl	%edx, %eax
	addl	$2, %eax
	movl	%eax, -140(%rbp)
	movl	-140(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -104(%rbp)
	movq	path(%rip), %rdx
	movq	-104(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movw	$47, (%rax)
	movq	-120(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-104(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	-104(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -136(%rbp)
	movl	-136(%rbp), %eax
	movl	%eax, -164(%rbp)
	movq	-112(%rbp), %rax
	movl	$438, %edx
	movl	$577, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movl	%eax, -160(%rbp)
	movq	$2, -96(%rbp)
	jmp	.L115
.L95:
	movl	24+data(%rip), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L116
	movq	$1, -96(%rbp)
	jmp	.L115
.L116:
	movq	$22, -96(%rbp)
	jmp	.L115
.L77:
	movq	nrd(%rip), %rax
	testq	%rax, %rax
	jle	.L118
	movq	$34, -96(%rbp)
	jmp	.L115
.L118:
	movq	$17, -96(%rbp)
	jmp	.L115
.L110:
	cmpl	$-1, -156(%rbp)
	jne	.L120
	movq	$78, -96(%rbp)
	jmp	.L115
.L120:
	movq	$85, -96(%rbp)
	jmp	.L115
.L67:
	movq	-208(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -176(%rbp)
	movq	$61, -96(%rbp)
	jmp	.L115
.L51:
	cmpl	$0, -172(%rbp)
	jne	.L122
	movq	$97, -96(%rbp)
	jmp	.L115
.L122:
	movq	$53, -96(%rbp)
	jmp	.L115
.L45:
	movl	-176(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	flushBuffer
	movq	$54, -96(%rbp)
	jmp	.L115
.L73:
	movl	$1, %eax
	jmp	.L124
.L52:
	movl	err(%rip), %eax
	testl	%eax, %eax
	je	.L125
	movq	$26, -96(%rbp)
	jmp	.L115
.L125:
	movq	$64, -96(%rbp)
	jmp	.L115
.L91:
	movq	filename(%rip), %rax
	movl	$0, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, open_file(%rip)
	movq	$76, -96(%rbp)
	jmp	.L115
.L104:
	movl	$1, %eax
	jmp	.L124
.L60:
	movq	-120(%rbp), %rax
	movzbl	18(%rax), %eax
	cmpb	$4, %al
	jne	.L127
	movq	$97, -96(%rbp)
	jmp	.L115
.L127:
	movq	$50, -96(%rbp)
	jmp	.L115
.L106:
	movq	buffer(%rip), %rcx
	movl	open_file(%rip), %eax
	movl	$64, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, bytes_read(%rip)
	movq	$65, -96(%rbp)
	jmp	.L115
.L42:
	movl	$0, %eax
	jmp	.L124
.L82:
	movq	-208(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, path(%rip)
	movq	path(%rip), %rax
	leaq	data(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, err(%rip)
	movq	$79, -96(%rbp)
	jmp	.L115
.L75:
	cmpl	$1, -180(%rbp)
	jne	.L129
	movq	$31, -96(%rbp)
	jmp	.L115
.L129:
	movq	$94, -96(%rbp)
	jmp	.L115
.L53:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$24, -96(%rbp)
	jmp	.L115
.L113:
	movq	-208(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, filename(%rip)
	movl	$1, -180(%rbp)
	movq	$54, -96(%rbp)
	jmp	.L115
.L97:
	movl	-176(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	flushBuffer
	movl	open_file(%rip), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	%eax, close_status(%rip)
	movq	$13, -96(%rbp)
	jmp	.L115
.L111:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$19, -96(%rbp)
	jmp	.L115
.L102:
	movq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	buffer(%rip), %rax
	movb	$0, (%rax)
	movq	$0, 8+sb(%rip)
	movq	16+sb(%rip), %rax
	movb	$0, (%rax)
	movl	$0, accumulator(%rip)
	movl	$0, first_text_found(%rip)
	movq	$97, -96(%rbp)
	jmp	.L115
.L96:
	movl	$1, %eax
	jmp	.L124
.L43:
	cmpl	$2, -180(%rbp)
	jne	.L131
	movq	$38, -96(%rbp)
	jmp	.L115
.L131:
	movq	$73, -96(%rbp)
	jmp	.L115
.L54:
	movl	open_file(%rip), %eax
	cmpl	$-1, %eax
	jne	.L133
	movq	$51, -96(%rbp)
	jmp	.L115
.L133:
	movq	$8, -96(%rbp)
	jmp	.L115
.L72:
	movl	$1, %eax
	jmp	.L124
.L61:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$41, -96(%rbp)
	jmp	.L115
.L48:
	movl	-160(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	%eax, -152(%rbp)
	movq	$67, -96(%rbp)
	jmp	.L115
.L94:
	movq	path(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$42, -96(%rbp)
	jmp	.L115
.L103:
	movl	close_status(%rip), %eax
	cmpl	$-1, %eax
	jne	.L135
	movq	$58, -96(%rbp)
	jmp	.L115
.L135:
	movq	$94, -96(%rbp)
	jmp	.L115
.L66:
	cmpl	$3, -196(%rbp)
	jne	.L137
	movq	$45, -96(%rbp)
	jmp	.L115
.L137:
	movq	$64, -96(%rbp)
	jmp	.L115
.L78:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$56, -96(%rbp)
	jmp	.L115
.L99:
	movl	$1, %eax
	jmp	.L124
.L101:
	movl	-160(%rbp), %edx
	movl	-176(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	flushBuffer
	movl	-164(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	%eax, -156(%rbp)
	movq	$4, -96(%rbp)
	jmp	.L115
.L86:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$46, -96(%rbp)
	jmp	.L115
.L62:
	cmpl	$-1, -152(%rbp)
	jne	.L139
	movq	$40, -96(%rbp)
	jmp	.L115
.L139:
	movq	$16, -96(%rbp)
	jmp	.L115
.L74:
	movq	-120(%rbp), %rax
	addq	$19, %rax
	movzbl	(%rax), %eax
	movzbl	.LC6(%rip), %edx
	movzbl	%al, %eax
	movzbl	%dl, %edx
	subl	%edx, %eax
	movl	%eax, -172(%rbp)
	movq	$82, -96(%rbp)
	jmp	.L115
.L69:
	movq	$44, -96(%rbp)
	jmp	.L115
.L70:
	movl	$1, %eax
	jmp	.L124
.L108:
	cmpq	$0, -120(%rbp)
	je	.L141
	movq	$55, -96(%rbp)
	jmp	.L115
.L141:
	movq	$10, -96(%rbp)
	jmp	.L115
.L93:
	cmpl	$-1, -148(%rbp)
	jne	.L143
	movq	$71, -96(%rbp)
	jmp	.L115
.L143:
	movq	$73, -96(%rbp)
	jmp	.L115
.L87:
	movq	path(%rip), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -128(%rbp)
	movq	$37, -96(%rbp)
	jmp	.L115
.L68:
	cmpl	$0, -176(%rbp)
	jg	.L145
	movq	$68, -96(%rbp)
	jmp	.L115
.L145:
	movq	$63, -96(%rbp)
	jmp	.L115
.L47:
	movl	bytes_read(%rip), %eax
	movl	-176(%rbp), %ecx
	movl	$1, %edx
	movl	%ecx, %esi
	movl	%eax, %edi
	call	wrap
	movq	buffer(%rip), %rax
	movl	$64, %edx
	movq	%rax, %rsi
	movl	$0, %edi
	call	read@PLT
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	movl	%eax, bytes_read(%rip)
	movq	$48, -96(%rbp)
	jmp	.L115
.L71:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$12, -96(%rbp)
	jmp	.L115
.L49:
	movl	bytes_read(%rip), %eax
	movl	-176(%rbp), %ecx
	movl	$1, %edx
	movl	%ecx, %esi
	movl	%eax, %edi
	call	wrap
	movq	buffer(%rip), %rcx
	movl	open_file(%rip), %eax
	movl	$64, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movl	%eax, bytes_read(%rip)
	movq	$65, -96(%rbp)
	jmp	.L115
.L90:
	movq	nrd(%rip), %rax
	movl	%eax, %ecx
	movl	-160(%rbp), %edx
	movl	-176(%rbp), %eax
	movl	%eax, %esi
	movl	%ecx, %edi
	call	wrap
	movq	buffer(%rip), %rcx
	movl	-164(%rbp), %eax
	movl	$64, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, nrd(%rip)
	movq	$52, -96(%rbp)
	jmp	.L115
.L56:
	movq	buffer(%rip), %rax
	movl	$64, %edx
	movq	%rax, %rsi
	movl	$0, %edi
	call	read@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movl	%eax, bytes_read(%rip)
	movq	$48, -96(%rbp)
	jmp	.L115
.L55:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$92, -96(%rbp)
	jmp	.L115
.L80:
	movl	bytes_read(%rip), %eax
	testl	%eax, %eax
	jle	.L147
	movq	$87, -96(%rbp)
	jmp	.L115
.L147:
	movq	$89, -96(%rbp)
	jmp	.L115
.L59:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$7, -96(%rbp)
	jmp	.L115
.L98:
	movl	24+data(%rip), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L149
	movq	$28, -96(%rbp)
	jmp	.L115
.L149:
	movq	$54, -96(%rbp)
	jmp	.L115
.L92:
	movl	$2, -180(%rbp)
	movq	$54, -96(%rbp)
	jmp	.L115
.L76:
	movq	-120(%rbp), %rax
	addq	$19, %rax
	movl	$5, %edx
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -168(%rbp)
	movq	$83, -96(%rbp)
	jmp	.L115
.L64:
	movl	bytes_read(%rip), %eax
	testl	%eax, %eax
	jle	.L151
	movq	$84, -96(%rbp)
	jmp	.L115
.L151:
	movq	$23, -96(%rbp)
	jmp	.L115
.L57:
	leaq	sb(%rip), %rax
	movq	%rax, %rdi
	call	strbuf_destroy
	movq	buffer(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -96(%rbp)
	jmp	.L115
.L83:
	movl	$0, -180(%rbp)
	movl	$64, %esi
	leaq	sb(%rip), %rax
	movq	%rax, %rdi
	call	strbuf_init
	movl	$64, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, buffer(%rip)
	movq	$35, -96(%rbp)
	jmp	.L115
.L109:
	movl	exit_status(%rip), %eax
	cmpl	$1, %eax
	jne	.L153
	movq	$57, -96(%rbp)
	jmp	.L115
.L153:
	movq	$96, -96(%rbp)
	jmp	.L115
.L40:
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -120(%rbp)
	movq	$6, -96(%rbp)
	jmp	.L115
.L58:
	movl	$0, first_text_found(%rip)
	movl	$0, space_ct(%rip)
	movl	$0, newline_ct(%rip)
	movq	buffer(%rip), %rcx
	movl	-164(%rbp), %eax
	movl	$64, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, nrd(%rip)
	movq	$52, -96(%rbp)
	jmp	.L115
.L88:
	cmpq	$0, -128(%rbp)
	je	.L155
	movq	$97, -96(%rbp)
	jmp	.L115
.L155:
	movq	$3, -96(%rbp)
	jmp	.L115
.L65:
	cmpl	$2, -196(%rbp)
	jne	.L157
	movq	$74, -96(%rbp)
	jmp	.L115
.L157:
	movq	$25, -96(%rbp)
	jmp	.L115
.L85:
	movl	$1, %eax
	jmp	.L124
.L44:
	movl	$1, %eax
	jmp	.L124
.L105:
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movl	%eax, -148(%rbp)
	movq	$27, -96(%rbp)
	jmp	.L115
.L84:
	movl	$1, %eax
	jmp	.L124
.L114:
	movl	$1, %eax
	jmp	.L124
.L81:
	movl	$1, %eax
	jmp	.L124
.L63:
	cmpl	$-1, -160(%rbp)
	jne	.L159
	movq	$75, -96(%rbp)
	jmp	.L115
.L159:
	movq	$72, -96(%rbp)
	jmp	.L115
.L50:
	cmpl	$0, -168(%rbp)
	jne	.L161
	movq	$97, -96(%rbp)
	jmp	.L115
.L161:
	movq	$69, -96(%rbp)
	jmp	.L115
.L107:
	movl	$1, %eax
	jmp	.L124
.L46:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -96(%rbp)
	jmp	.L115
.L89:
	cmpl	$1, -196(%rbp)
	jle	.L163
	movq	$62, -96(%rbp)
	jmp	.L115
.L163:
	movq	$88, -96(%rbp)
	jmp	.L115
.L112:
	cmpl	$-1, -164(%rbp)
	jne	.L165
	movq	$18, -96(%rbp)
	jmp	.L115
.L165:
	movq	$66, -96(%rbp)
	jmp	.L115
.L168:
	nop
.L115:
	jmp	.L167
.L124:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC12:
	.string	"\n"
.LC13:
	.string	" "
	.text
	.globl	flushBuffer
	.type	flushBuffer, @function
flushBuffer:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L195:
	cmpq	$14, -8(%rbp)
	ja	.L196
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L172(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L172(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L172:
	.long	.L182-.L172
	.long	.L181-.L172
	.long	.L196-.L172
	.long	.L196-.L172
	.long	.L180-.L172
	.long	.L179-.L172
	.long	.L178-.L172
	.long	.L196-.L172
	.long	.L177-.L172
	.long	.L176-.L172
	.long	.L175-.L172
	.long	.L174-.L172
	.long	.L173-.L172
	.long	.L196-.L172
	.long	.L197-.L172
	.text
.L180:
	movl	accumulator(%rip), %eax
	testl	%eax, %eax
	je	.L183
	movq	$5, -8(%rbp)
	jmp	.L185
.L183:
	movq	$6, -8(%rbp)
	jmp	.L185
.L173:
	movl	-24(%rbp), %eax
	movl	$1, %edx
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$14, -8(%rbp)
	jmp	.L185
.L177:
	movl	-24(%rbp), %eax
	movl	$1, %edx
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	$0, accumulator(%rip)
	movq	$6, -8(%rbp)
	jmp	.L185
.L181:
	movq	8+sb(%rip), %rdx
	movl	-20(%rbp), %eax
	cltq
	cmpq	%rax, %rdx
	jbe	.L187
	movq	$10, -8(%rbp)
	jmp	.L185
.L187:
	movq	$12, -8(%rbp)
	jmp	.L185
.L174:
	movl	accumulator(%rip), %eax
	addl	$1, %eax
	movslq	%eax, %rdx
	movq	8+sb(%rip), %rax
	addq	%rax, %rdx
	movl	-20(%rbp), %eax
	cltq
	cmpq	%rax, %rdx
	jbe	.L189
	movq	$9, -8(%rbp)
	jmp	.L185
.L189:
	movq	$4, -8(%rbp)
	jmp	.L185
.L176:
	movl	accumulator(%rip), %eax
	testl	%eax, %eax
	je	.L191
	movq	$8, -8(%rbp)
	jmp	.L185
.L191:
	movq	$4, -8(%rbp)
	jmp	.L185
.L178:
	movq	8+sb(%rip), %rdx
	movq	16+sb(%rip), %rcx
	movl	-24(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	8+sb(%rip), %rax
	movl	%eax, %edx
	movl	accumulator(%rip), %eax
	addl	%edx, %eax
	movl	%eax, accumulator(%rip)
	movq	$1, -8(%rbp)
	jmp	.L185
.L179:
	movl	-24(%rbp), %eax
	movl	$1, %edx
	leaq	.LC13(%rip), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	accumulator(%rip), %eax
	addl	$1, %eax
	movl	%eax, accumulator(%rip)
	movq	$6, -8(%rbp)
	jmp	.L185
.L175:
	movl	$1, exit_status(%rip)
	movq	$12, -8(%rbp)
	jmp	.L185
.L182:
	movq	8+sb(%rip), %rax
	testq	%rax, %rax
	je	.L193
	movq	$11, -8(%rbp)
	jmp	.L185
.L193:
	movq	$12, -8(%rbp)
	jmp	.L185
.L196:
	nop
.L185:
	jmp	.L195
.L197:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	flushBuffer, .-flushBuffer
	.globl	strbuf_destroy
	.type	strbuf_destroy, @function
strbuf_destroy:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L203:
	cmpq	$0, -8(%rbp)
	je	.L204
	cmpq	$1, -8(%rbp)
	jne	.L205
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -8(%rbp)
	jmp	.L201
.L205:
	nop
.L201:
	jmp	.L203
.L204:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	strbuf_destroy, .-strbuf_destroy
	.globl	strbuf_init
	.type	strbuf_init, @function
strbuf_init:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$2, -16(%rbp)
.L219:
	cmpq	$6, -16(%rbp)
	ja	.L220
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L209(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L209(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L209:
	.long	.L214-.L209
	.long	.L213-.L209
	.long	.L212-.L209
	.long	.L220-.L209
	.long	.L211-.L209
	.long	.L210-.L209
	.long	.L208-.L209
	.text
.L211:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$0, -16(%rbp)
	jmp	.L215
.L213:
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$6, -16(%rbp)
	jmp	.L215
.L208:
	movl	$0, %eax
	jmp	.L216
.L210:
	movl	$1, %eax
	jmp	.L216
.L214:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	jne	.L217
	movq	$5, -16(%rbp)
	jmp	.L215
.L217:
	movq	$1, -16(%rbp)
	jmp	.L215
.L212:
	movq	$4, -16(%rbp)
	jmp	.L215
.L220:
	nop
.L215:
	jmp	.L219
.L216:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	strbuf_init, .-strbuf_init
	.section	.rodata
.LC14:
	.string	"\n\n"
	.text
	.globl	wrap
	.type	wrap, @function
wrap:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movl	%edx, -44(%rbp)
	movq	$33, -8(%rbp)
.L276:
	cmpq	$33, -8(%rbp)
	ja	.L278
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L224(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L224(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L224:
	.long	.L249-.L224
	.long	.L278-.L224
	.long	.L248-.L224
	.long	.L247-.L224
	.long	.L246-.L224
	.long	.L245-.L224
	.long	.L278-.L224
	.long	.L278-.L224
	.long	.L278-.L224
	.long	.L278-.L224
	.long	.L244-.L224
	.long	.L243-.L224
	.long	.L242-.L224
	.long	.L241-.L224
	.long	.L240-.L224
	.long	.L239-.L224
	.long	.L238-.L224
	.long	.L237-.L224
	.long	.L236-.L224
	.long	.L235-.L224
	.long	.L278-.L224
	.long	.L234-.L224
	.long	.L233-.L224
	.long	.L232-.L224
	.long	.L278-.L224
	.long	.L231-.L224
	.long	.L278-.L224
	.long	.L230-.L224
	.long	.L229-.L224
	.long	.L228-.L224
	.long	.L227-.L224
	.long	.L226-.L224
	.long	.L225-.L224
	.long	.L223-.L224
	.text
.L236:
	movq	8+sb(%rip), %rdx
	movq	16+sb(%rip), %rcx
	movl	-44(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	8+sb(%rip), %rax
	movl	%eax, %edx
	movl	accumulator(%rip), %eax
	addl	%edx, %eax
	movl	%eax, accumulator(%rip)
	movq	$28, -8(%rbp)
	jmp	.L250
.L231:
	movl	accumulator(%rip), %eax
	addl	$1, %eax
	movslq	%eax, %rdx
	movq	8+sb(%rip), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	cmpq	%rax, %rdx
	jbe	.L251
	movq	$21, -8(%rbp)
	jmp	.L250
.L251:
	movq	$27, -8(%rbp)
	jmp	.L250
.L246:
	movl	$1, exit_status(%rip)
	movq	$19, -8(%rbp)
	jmp	.L250
.L227:
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L250
.L240:
	movl	$1, first_text_found(%rip)
	movq	$13, -8(%rbp)
	jmp	.L250
.L239:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	buffer(%rip), %rcx
	movl	-20(%rbp), %eax
	cltq
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L253
	movq	$0, -8(%rbp)
	jmp	.L250
.L253:
	movq	$11, -8(%rbp)
	jmp	.L250
.L226:
	movl	-44(%rbp), %eax
	movl	$1, %edx
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	$0, accumulator(%rip)
	movq	$18, -8(%rbp)
	jmp	.L250
.L242:
	movl	space_ct(%rip), %eax
	addl	$1, %eax
	movl	%eax, space_ct(%rip)
	movq	$13, -8(%rbp)
	jmp	.L250
.L232:
	movl	first_text_found(%rip), %eax
	testl	%eax, %eax
	jne	.L255
	movq	$14, -8(%rbp)
	jmp	.L250
.L255:
	movq	$13, -8(%rbp)
	jmp	.L250
.L247:
	movl	newline_ct(%rip), %eax
	testl	%eax, %eax
	jne	.L257
	movq	$16, -8(%rbp)
	jmp	.L250
.L257:
	movq	$10, -8(%rbp)
	jmp	.L250
.L238:
	movl	space_ct(%rip), %eax
	testl	%eax, %eax
	jne	.L259
	movq	$25, -8(%rbp)
	jmp	.L250
.L259:
	movq	$10, -8(%rbp)
	jmp	.L250
.L234:
	movl	accumulator(%rip), %eax
	testl	%eax, %eax
	je	.L261
	movq	$31, -8(%rbp)
	jmp	.L250
.L261:
	movq	$27, -8(%rbp)
	jmp	.L250
.L243:
	movl	newline_ct(%rip), %eax
	cmpl	$1, %eax
	jle	.L263
	movq	$2, -8(%rbp)
	jmp	.L250
.L263:
	movq	$17, -8(%rbp)
	jmp	.L250
.L241:
	addl	$1, -20(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L250
.L235:
	movq	$0, 8+sb(%rip)
	movq	16+sb(%rip), %rax
	movb	$0, (%rax)
	movq	$10, -8(%rbp)
	jmp	.L250
.L225:
	movl	-20(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jge	.L265
	movq	$30, -8(%rbp)
	jmp	.L250
.L265:
	movq	$5, -8(%rbp)
	jmp	.L250
.L237:
	movq	buffer(%rip), %rdx
	movl	-20(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	sb(%rip), %rax
	movq	%rax, %rdi
	call	strbuf_append
	movl	$0, space_ct(%rip)
	movl	$0, newline_ct(%rip)
	movq	$23, -8(%rbp)
	jmp	.L250
.L230:
	movl	accumulator(%rip), %eax
	testl	%eax, %eax
	je	.L267
	movq	$29, -8(%rbp)
	jmp	.L250
.L267:
	movq	$18, -8(%rbp)
	jmp	.L250
.L233:
	movl	newline_ct(%rip), %eax
	addl	$1, %eax
	movl	%eax, newline_ct(%rip)
	movq	$13, -8(%rbp)
	jmp	.L250
.L229:
	movq	8+sb(%rip), %rdx
	movl	-40(%rbp), %eax
	cltq
	cmpq	%rax, %rdx
	jbe	.L269
	movq	$4, -8(%rbp)
	jmp	.L250
.L269:
	movq	$19, -8(%rbp)
	jmp	.L250
.L245:
	movl	$0, %eax
	jmp	.L277
.L223:
	movl	$0, -20(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L250
.L244:
	movq	buffer(%rip), %rdx
	movl	-20(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	jne	.L272
	movq	$22, -8(%rbp)
	jmp	.L250
.L272:
	movq	$12, -8(%rbp)
	jmp	.L250
.L249:
	movl	first_text_found(%rip), %eax
	testl	%eax, %eax
	je	.L274
	movq	$3, -8(%rbp)
	jmp	.L250
.L274:
	movq	$13, -8(%rbp)
	jmp	.L250
.L228:
	movl	-44(%rbp), %eax
	movl	$1, %edx
	leaq	.LC13(%rip), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	accumulator(%rip), %eax
	addl	$1, %eax
	movl	%eax, accumulator(%rip)
	movq	$18, -8(%rbp)
	jmp	.L250
.L248:
	movl	-44(%rbp), %eax
	movl	$2, %edx
	leaq	.LC14(%rip), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	$0, accumulator(%rip)
	movq	$17, -8(%rbp)
	jmp	.L250
.L278:
	nop
.L250:
	jmp	.L276
.L277:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	wrap, .-wrap
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
