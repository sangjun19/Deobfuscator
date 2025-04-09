	.file	"Fedjmike_mini-c_cc_flatten.c"
	.text
	.globl	is_fn
	.bss
	.align 8
	.type	is_fn, @object
	.size	is_fn, 8
is_fn:
	.zero	8
	.globl	curln
	.align 4
	.type	curln, @object
	.size	curln, 4
curln:
	.zero	4
	.globl	inputname
	.align 8
	.type	inputname, @object
	.size	inputname, 8
inputname:
	.zero	8
	.globl	offsets
	.align 8
	.type	offsets, @object
	.size	offsets, 8
offsets:
	.zero	8
	.globl	token_ident
	.align 4
	.type	token_ident, @object
	.size	token_ident, 4
token_ident:
	.zero	4
	.globl	decl_param
	.align 4
	.type	decl_param, @object
	.size	decl_param, 4
decl_param:
	.zero	4
	.globl	return_to
	.align 4
	.type	return_to, @object
	.size	return_to, 4
return_to:
	.zero	4
	.globl	errors
	.align 4
	.type	errors, @object
	.size	errors, 4
errors:
	.zero	4
	.globl	output
	.align 8
	.type	output, @object
	.size	output, 8
output:
	.zero	8
	.globl	curch
	.type	curch, @object
	.size	curch, 1
curch:
	.zero	1
	.globl	input
	.align 8
	.type	input, @object
	.size	input, 8
input:
	.zero	8
	.globl	locals
	.align 8
	.type	locals, @object
	.size	locals, 8
locals:
	.zero	8
	.globl	word_size
	.align 4
	.type	word_size, @object
	.size	word_size, 4
word_size:
	.zero	4
	.globl	label_no
	.align 4
	.type	label_no, @object
	.size	label_no, 4
label_no:
	.zero	4
	.globl	token_other
	.align 4
	.type	token_other, @object
	.size	token_other, 4
token_other:
	.zero	4
	.globl	global_no
	.align 4
	.type	global_no, @object
	.size	global_no, 4
global_no:
	.zero	4
	.globl	globals
	.align 8
	.type	globals, @object
	.size	globals, 8
globals:
	.zero	8
	.globl	ptr_size
	.align 4
	.type	ptr_size, @object
	.size	ptr_size, 4
ptr_size:
	.zero	4
	.globl	param_no
	.align 4
	.type	param_no, @object
	.size	param_no, 4
param_no:
	.zero	4
	.globl	token_char
	.align 4
	.type	token_char, @object
	.size	token_char, 4
token_char:
	.zero	4
	.globl	buflength
	.align 4
	.type	buflength, @object
	.size	buflength, 4
buflength:
	.zero	4
	.globl	lvalue
	.type	lvalue, @object
	.size	lvalue, 1
lvalue:
	.zero	1
	.globl	_TIG_IZ_BYwQ_argv
	.align 8
	.type	_TIG_IZ_BYwQ_argv, @object
	.size	_TIG_IZ_BYwQ_argv, 8
_TIG_IZ_BYwQ_argv:
	.zero	8
	.globl	_TIG_IZ_BYwQ_envp
	.align 8
	.type	_TIG_IZ_BYwQ_envp, @object
	.size	_TIG_IZ_BYwQ_envp, 8
_TIG_IZ_BYwQ_envp:
	.zero	8
	.globl	local_no
	.align 4
	.type	local_no, @object
	.size	local_no, 4
local_no:
	.zero	4
	.globl	decl_module
	.align 4
	.type	decl_module, @object
	.size	decl_module, 4
decl_module:
	.zero	4
	.globl	token_int
	.align 4
	.type	token_int, @object
	.size	token_int, 4
token_int:
	.zero	4
	.globl	token
	.align 4
	.type	token, @object
	.size	token, 4
token:
	.zero	4
	.globl	token_str
	.align 4
	.type	token_str, @object
	.size	token_str, 4
token_str:
	.zero	4
	.globl	decl_local
	.align 4
	.type	decl_local, @object
	.size	decl_local, 4
decl_local:
	.zero	4
	.globl	_TIG_IZ_BYwQ_argc
	.align 4
	.type	_TIG_IZ_BYwQ_argc, @object
	.size	_TIG_IZ_BYwQ_argc, 4
_TIG_IZ_BYwQ_argc:
	.zero	4
	.globl	buffer
	.align 8
	.type	buffer, @object
	.size	buffer, 8
buffer:
	.zero	8
	.text
	.globl	new_param
	.type	new_param, @function
new_param:
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
	movq	$1, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movq	$0, -8(%rbp)
	jmp	.L5
.L4:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	new_local
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	param_no(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	param_no(%rip), %eax
	addl	$1, %eax
	movl	%eax, param_no(%rip)
	movl	-12(%rbp), %eax
	leal	2(%rax), %ecx
	movl	word_size(%rip), %eax
	movq	offsets(%rip), %rsi
	movl	-16(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	addq	%rsi, %rdx
	imull	%ecx, %eax
	movl	%eax, (%rdx)
	movq	$2, -8(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	new_param, .-new_param
	.section	.rodata
.LC0:
	.string	"if"
.LC1:
	.string	"("
.LC2:
	.string	")"
	.text
	.globl	if_branch
	.type	if_branch, @function
if_branch:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L16:
	cmpq	$2, -8(%rbp)
	je	.L17
	cmpq	$2, -8(%rbp)
	ja	.L18
	cmpq	$0, -8(%rbp)
	je	.L13
	cmpq	$1, -8(%rbp)
	jne	.L18
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	match
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	match
	movl	$0, %edi
	call	expr
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	match
	movl	$0, %edi
	call	branch
	movq	$2, -8(%rbp)
	jmp	.L14
.L13:
	movq	$1, -8(%rbp)
	jmp	.L14
.L18:
	nop
.L14:
	jmp	.L16
.L17:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	if_branch, .-if_branch
	.section	.rodata
.LC3:
	.string	"\t_%08d:\n"
.LC4:
	.string	"mov esp, ebp\npop ebp\nret\n"
.LC5:
	.string	".globl %s\n%s:\n"
	.align 8
.LC6:
	.string	"push ebp\nmov ebp, esp\nsub esp, %d\njmp _%08d\n"
	.text
	.globl	function
	.type	function, @function
function:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$0, -8(%rbp)
.L25:
	cmpq	$2, -8(%rbp)
	je	.L20
	cmpq	$2, -8(%rbp)
	ja	.L27
	cmpq	$0, -8(%rbp)
	je	.L22
	cmpq	$1, -8(%rbp)
	jne	.L27
	jmp	.L26
.L22:
	movq	$2, -8(%rbp)
	jmp	.L24
.L20:
	call	new_label
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	emit_label
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	call	new_label
	movl	%eax, return_to(%rip)
	call	line
	movl	return_to(%rip), %edx
	movq	output(%rip), %rax
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$25, %edx
	movl	$1, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	output(%rip), %rax
	movq	-40(%rbp), %rcx
	movq	-40(%rbp), %rdx
	leaq	.LC5(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	local_no(%rip), %edx
	movl	word_size(%rip), %eax
	imull	%eax, %edx
	movq	output(%rip), %rax
	movl	-12(%rbp), %ecx
	leaq	.LC6(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$1, -8(%rbp)
	jmp	.L24
.L27:
	nop
.L24:
	jmp	.L25
.L26:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	function, .-function
	.globl	new_local
	.type	new_local, @function
new_local:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L34:
	cmpq	$2, -8(%rbp)
	je	.L29
	cmpq	$2, -8(%rbp)
	ja	.L36
	cmpq	$0, -8(%rbp)
	je	.L31
	cmpq	$1, -8(%rbp)
	jne	.L36
	movl	local_no(%rip), %eax
	movl	param_no(%rip), %edx
	subl	%edx, %eax
	movl	%eax, -12(%rbp)
	movq	locals(%rip), %rdx
	movl	local_no(%rip), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rax, (%rdx)
	movl	word_size(%rip), %eax
	negl	%eax
	movl	%eax, %esi
	movl	-12(%rbp), %eax
	leal	1(%rax), %ecx
	movq	offsets(%rip), %rdx
	movl	local_no(%rip), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	%esi, %eax
	imull	%ecx, %eax
	movl	%eax, (%rdx)
	movl	local_no(%rip), %eax
	movl	%eax, -16(%rbp)
	movl	local_no(%rip), %eax
	addl	$1, %eax
	movl	%eax, local_no(%rip)
	movq	$0, -8(%rbp)
	jmp	.L32
.L31:
	movl	-16(%rbp), %eax
	jmp	.L35
.L29:
	movq	$1, -8(%rbp)
	jmp	.L32
.L36:
	nop
.L32:
	jmp	.L34
.L35:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	new_local, .-new_local
	.globl	next
	.type	next, @function
next:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	$44, -8(%rbp)
.L192:
	cmpq	$100, -8(%rbp)
	ja	.L193
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L40(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L40(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L40:
	.long	.L115-.L40
	.long	.L114-.L40
	.long	.L113-.L40
	.long	.L112-.L40
	.long	.L111-.L40
	.long	.L110-.L40
	.long	.L109-.L40
	.long	.L193-.L40
	.long	.L108-.L40
	.long	.L107-.L40
	.long	.L106-.L40
	.long	.L105-.L40
	.long	.L104-.L40
	.long	.L103-.L40
	.long	.L102-.L40
	.long	.L101-.L40
	.long	.L100-.L40
	.long	.L99-.L40
	.long	.L98-.L40
	.long	.L97-.L40
	.long	.L96-.L40
	.long	.L95-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L94-.L40
	.long	.L93-.L40
	.long	.L92-.L40
	.long	.L91-.L40
	.long	.L90-.L40
	.long	.L89-.L40
	.long	.L88-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L87-.L40
	.long	.L86-.L40
	.long	.L85-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L84-.L40
	.long	.L83-.L40
	.long	.L82-.L40
	.long	.L194-.L40
	.long	.L80-.L40
	.long	.L193-.L40
	.long	.L79-.L40
	.long	.L78-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L77-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L76-.L40
	.long	.L75-.L40
	.long	.L74-.L40
	.long	.L73-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L72-.L40
	.long	.L71-.L40
	.long	.L193-.L40
	.long	.L70-.L40
	.long	.L69-.L40
	.long	.L193-.L40
	.long	.L68-.L40
	.long	.L67-.L40
	.long	.L66-.L40
	.long	.L65-.L40
	.long	.L64-.L40
	.long	.L63-.L40
	.long	.L62-.L40
	.long	.L61-.L40
	.long	.L60-.L40
	.long	.L59-.L40
	.long	.L193-.L40
	.long	.L58-.L40
	.long	.L57-.L40
	.long	.L56-.L40
	.long	.L55-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L54-.L40
	.long	.L53-.L40
	.long	.L52-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L51-.L40
	.long	.L50-.L40
	.long	.L49-.L40
	.long	.L193-.L40
	.long	.L48-.L40
	.long	.L47-.L40
	.long	.L46-.L40
	.long	.L193-.L40
	.long	.L193-.L40
	.long	.L45-.L40
	.long	.L44-.L40
	.long	.L43-.L40
	.long	.L193-.L40
	.long	.L42-.L40
	.long	.L194-.L40
	.long	.L39-.L40
	.text
.L98:
	movl	$0, -72(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L116
.L54:
	call	__ctype_b_loc@PLT
	movq	%rax, -40(%rbp)
	movq	$57, -8(%rbp)
	jmp	.L116
.L93:
	movl	token_char(%rip), %eax
	movl	%eax, token(%rip)
	movq	$17, -8(%rbp)
	jmp	.L116
.L75:
	cmpb	$47, -83(%rbp)
	jne	.L117
	movq	$89, -8(%rbp)
	jmp	.L116
.L117:
	movq	$34, -8(%rbp)
	jmp	.L116
.L111:
	movzbl	curch(%rip), %eax
	cmpb	$43, %al
	jne	.L119
	movq	$30, -8(%rbp)
	jmp	.L116
.L119:
	movq	$53, -8(%rbp)
	jmp	.L116
.L88:
	call	eat_char
	movq	$85, -8(%rbp)
	jmp	.L116
.L102:
	cmpl	$0, -56(%rbp)
	je	.L121
	movq	$2, -8(%rbp)
	jmp	.L116
.L121:
	movq	$75, -8(%rbp)
	jmp	.L116
.L101:
	movl	buflength(%rip), %eax
	movl	%eax, -52(%rbp)
	movl	buflength(%rip), %eax
	addl	$1, %eax
	movl	%eax, buflength(%rip)
	movq	buffer(%rip), %rdx
	movl	-52(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$41, -8(%rbp)
	jmp	.L116
.L52:
	call	eat_char
	movq	$76, -8(%rbp)
	jmp	.L116
.L48:
	movzbl	curch(%rip), %eax
	cmpb	$10, %al
	je	.L123
	movq	$40, -8(%rbp)
	jmp	.L116
.L123:
	movq	$65, -8(%rbp)
	jmp	.L116
.L104:
	call	next_char
	movq	$89, -8(%rbp)
	jmp	.L116
.L62:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	curch(%rip), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L125
	movq	$45, -8(%rbp)
	jmp	.L116
.L125:
	movq	$9, -8(%rbp)
	jmp	.L116
.L108:
	movzbl	curch(%rip), %eax
	cmpb	$62, %al
	jne	.L127
	movq	$30, -8(%rbp)
	jmp	.L116
.L127:
	movq	$86, -8(%rbp)
	jmp	.L116
.L43:
	movzbl	curch(%rip), %eax
	cmpb	$33, %al
	jne	.L129
	movq	$30, -8(%rbp)
	jmp	.L116
.L129:
	movq	$8, -8(%rbp)
	jmp	.L116
.L78:
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L116
.L73:
	movzbl	curch(%rip), %eax
	cmpb	$35, %al
	jne	.L131
	movq	$89, -8(%rbp)
	jmp	.L116
.L131:
	movq	$42, -8(%rbp)
	jmp	.L116
.L114:
	cmpl	$0, -60(%rbp)
	je	.L133
	movq	$82, -8(%rbp)
	jmp	.L116
.L133:
	movq	$15, -8(%rbp)
	jmp	.L116
.L53:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	curch(%rip), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L135
	movq	$74, -8(%rbp)
	jmp	.L116
.L135:
	movq	$6, -8(%rbp)
	jmp	.L116
.L55:
	movzbl	curch(%rip), %eax
	cmpb	$39, %al
	jne	.L137
	movq	$98, -8(%rbp)
	jmp	.L116
.L137:
	movq	$38, -8(%rbp)
	jmp	.L116
.L61:
	movzbl	curch(%rip), %eax
	cmpb	$33, %al
	je	.L139
	movq	$71, -8(%rbp)
	jmp	.L116
.L139:
	movq	$67, -8(%rbp)
	jmp	.L116
.L112:
	movl	-64(%rbp), %eax
	movl	%eax, -60(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L116
.L100:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	curch(%rip), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L141
	movq	$58, -8(%rbp)
	jmp	.L116
.L141:
	movq	$60, -8(%rbp)
	jmp	.L116
.L94:
	call	next_char
	movq	$68, -8(%rbp)
	jmp	.L116
.L95:
	cmpl	$0, -80(%rbp)
	je	.L143
	movq	$65, -8(%rbp)
	jmp	.L116
.L143:
	movq	$12, -8(%rbp)
	jmp	.L116
.L45:
	movzbl	curch(%rip), %eax
	cmpb	$10, %al
	jne	.L145
	movq	$24, -8(%rbp)
	jmp	.L116
.L145:
	movq	$20, -8(%rbp)
	jmp	.L116
.L56:
	movl	token(%rip), %edx
	movl	token_ident(%rip), %eax
	cmpl	%eax, %edx
	jne	.L147
	movq	$80, -8(%rbp)
	jmp	.L116
.L147:
	movq	$90, -8(%rbp)
	jmp	.L116
.L72:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	curch(%rip), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8, %eax
	testl	%eax, %eax
	je	.L149
	movq	$39, -8(%rbp)
	jmp	.L116
.L149:
	movq	$13, -8(%rbp)
	jmp	.L116
.L63:
	movzbl	curch(%rip), %eax
	cmpb	$32, %al
	jne	.L151
	movq	$24, -8(%rbp)
	jmp	.L116
.L151:
	movq	$66, -8(%rbp)
	jmp	.L116
.L51:
	movq	buffer(%rip), %rax
	movzbl	(%rax), %edx
	movzbl	curch(%rip), %eax
	cmpb	%al, %dl
	jne	.L153
	movq	$70, -8(%rbp)
	jmp	.L116
.L153:
	movq	$67, -8(%rbp)
	jmp	.L116
.L39:
	movl	$0, buflength(%rip)
	movl	token_other(%rip), %eax
	movl	%eax, token(%rip)
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$69, -8(%rbp)
	jmp	.L116
.L92:
	movl	$0, -64(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L116
.L42:
	movzbl	curch(%rip), %eax
	cmpb	$34, %al
	jne	.L155
	movq	$10, -8(%rbp)
	jmp	.L116
.L155:
	movq	$25, -8(%rbp)
	jmp	.L116
.L105:
	movzbl	curch(%rip), %eax
	cmpb	$38, %al
	jne	.L157
	movq	$30, -8(%rbp)
	jmp	.L116
.L157:
	movq	$28, -8(%rbp)
	jmp	.L116
.L107:
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L116
.L103:
	movzbl	curch(%rip), %eax
	cmpb	$95, %al
	jne	.L159
	movq	$39, -8(%rbp)
	jmp	.L116
.L159:
	movq	$63, -8(%rbp)
	jmp	.L116
.L68:
	movl	$0, -72(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L116
.L76:
	call	next_char
	movb	%al, -83(%rbp)
	movq	$52, -8(%rbp)
	jmp	.L116
.L97:
	movl	$1, -64(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L116
.L99:
	movzbl	curch(%rip), %eax
	movb	%al, -81(%rbp)
	call	eat_char
	movq	$91, -8(%rbp)
	jmp	.L116
.L47:
	call	__ctype_b_loc@PLT
	movq	%rax, -32(%rbp)
	movq	$81, -8(%rbp)
	jmp	.L116
.L82:
	movq	input(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -80(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L116
.L64:
	movzbl	curch(%rip), %eax
	cmpb	$61, %al
	jne	.L161
	movq	$95, -8(%rbp)
	jmp	.L116
.L161:
	movq	$15, -8(%rbp)
	jmp	.L116
.L70:
	movl	token_int(%rip), %eax
	movl	%eax, token(%rip)
	movq	$76, -8(%rbp)
	jmp	.L116
.L109:
	movl	$0, -64(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L116
.L91:
	cmpl	$0, -76(%rbp)
	je	.L163
	movq	$18, -8(%rbp)
	jmp	.L116
.L163:
	movq	$0, -8(%rbp)
	jmp	.L116
.L84:
	movzbl	curch(%rip), %eax
	cmpb	$34, %al
	jne	.L165
	movq	$98, -8(%rbp)
	jmp	.L116
.L165:
	movq	$4, -8(%rbp)
	jmp	.L116
.L69:
	call	eat_char
	movq	$48, -8(%rbp)
	jmp	.L116
.L49:
	movq	input(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -56(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L116
.L71:
	movl	token_ident(%rip), %eax
	movl	%eax, token(%rip)
	movq	$76, -8(%rbp)
	jmp	.L116
.L86:
	movl	$47, %edi
	call	prev_char
	movb	%al, -82(%rbp)
	movq	$72, -8(%rbp)
	jmp	.L116
.L58:
	movq	input(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -68(%rbp)
	movq	$64, -8(%rbp)
	jmp	.L116
.L57:
	movzbl	curch(%rip), %eax
	cmpb	$92, %al
	jne	.L167
	movq	$61, -8(%rbp)
	jmp	.L116
.L167:
	movq	$48, -8(%rbp)
	jmp	.L116
.L77:
	call	eat_char
	movq	$91, -8(%rbp)
	jmp	.L116
.L60:
	call	eat_char
	movq	$15, -8(%rbp)
	jmp	.L116
.L90:
	movzbl	curch(%rip), %eax
	cmpb	$61, %al
	jne	.L169
	movq	$30, -8(%rbp)
	jmp	.L116
.L169:
	movq	$96, -8(%rbp)
	jmp	.L116
.L74:
	movzbl	curch(%rip), %eax
	cmpb	$45, %al
	jne	.L171
	movq	$30, -8(%rbp)
	jmp	.L116
.L171:
	movq	$29, -8(%rbp)
	jmp	.L116
.L66:
	call	next
	movq	$99, -8(%rbp)
	jmp	.L116
.L79:
	movq	$68, -8(%rbp)
	jmp	.L116
.L110:
	movl	-72(%rbp), %eax
	movl	%eax, -60(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L116
.L46:
	movzbl	curch(%rip), %eax
	cmpb	%al, -81(%rbp)
	je	.L173
	movq	$87, -8(%rbp)
	jmp	.L116
.L173:
	movq	$2, -8(%rbp)
	jmp	.L116
.L59:
	cmpb	$0, -82(%rbp)
	je	.L175
	movq	$89, -8(%rbp)
	jmp	.L116
.L175:
	movq	$100, -8(%rbp)
	jmp	.L116
.L87:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	curch(%rip), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L178
	movq	$45, -8(%rbp)
	jmp	.L116
.L178:
	movq	$77, -8(%rbp)
	jmp	.L116
.L67:
	cmpl	$0, -68(%rbp)
	je	.L180
	movq	$26, -8(%rbp)
	jmp	.L116
.L180:
	movq	$19, -8(%rbp)
	jmp	.L116
.L44:
	call	eat_char
	movq	$15, -8(%rbp)
	jmp	.L116
.L106:
	movl	token_str(%rip), %eax
	movl	%eax, token(%rip)
	movq	$17, -8(%rbp)
	jmp	.L116
.L80:
	movzbl	curch(%rip), %eax
	cmpb	$47, %al
	jne	.L182
	movq	$51, -8(%rbp)
	jmp	.L116
.L182:
	movq	$100, -8(%rbp)
	jmp	.L116
.L115:
	movl	$1, -72(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L116
.L83:
	movq	input(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -76(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L116
.L65:
	movzbl	curch(%rip), %eax
	cmpb	$13, %al
	jne	.L184
	movq	$24, -8(%rbp)
	jmp	.L116
.L184:
	movq	$94, -8(%rbp)
	jmp	.L116
.L85:
	call	eat_char
	movq	$15, -8(%rbp)
	jmp	.L116
.L89:
	movzbl	curch(%rip), %eax
	cmpb	$124, %al
	jne	.L186
	movq	$30, -8(%rbp)
	jmp	.L116
.L186:
	movq	$11, -8(%rbp)
	jmp	.L116
.L50:
	movzbl	curch(%rip), %eax
	cmpb	$60, %al
	jne	.L188
	movq	$30, -8(%rbp)
	jmp	.L116
.L188:
	movq	$35, -8(%rbp)
	jmp	.L116
.L113:
	call	eat_char
	movq	$15, -8(%rbp)
	jmp	.L116
.L96:
	movzbl	curch(%rip), %eax
	cmpb	$9, %al
	jne	.L190
	movq	$24, -8(%rbp)
	jmp	.L116
.L190:
	movq	$54, -8(%rbp)
	jmp	.L116
.L193:
	nop
.L116:
	jmp	.L192
.L194:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	next, .-next
	.globl	see
	.type	see, @function
see:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L207:
	cmpq	$4, -8(%rbp)
	ja	.L209
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L198(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L198(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L198:
	.long	.L202-.L198
	.long	.L201-.L198
	.long	.L200-.L198
	.long	.L199-.L198
	.long	.L197-.L198
	.text
.L197:
	cmpl	$0, -12(%rbp)
	setne	%al
	jmp	.L208
.L201:
	cmpl	$0, -16(%rbp)
	je	.L204
	movq	$0, -8(%rbp)
	jmp	.L206
.L204:
	movq	$3, -8(%rbp)
	jmp	.L206
.L199:
	movl	$1, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L206
.L202:
	movl	$0, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L206
.L200:
	movq	buffer(%rip), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L206
.L209:
	nop
.L206:
	jmp	.L207
.L208:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	see, .-see
	.globl	new_label
	.type	new_label, @function
new_label:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L216:
	cmpq	$2, -8(%rbp)
	je	.L211
	cmpq	$2, -8(%rbp)
	ja	.L218
	cmpq	$0, -8(%rbp)
	je	.L213
	cmpq	$1, -8(%rbp)
	jne	.L218
	movl	-12(%rbp), %eax
	jmp	.L217
.L213:
	movq	$2, -8(%rbp)
	jmp	.L215
.L211:
	movl	label_no(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	label_no(%rip), %eax
	addl	$1, %eax
	movl	%eax, label_no(%rip)
	movq	$1, -8(%rbp)
	jmp	.L215
.L218:
	nop
.L215:
	jmp	.L216
.L217:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	new_label, .-new_label
	.section	.rodata
.LC7:
	.string	"<"
.LC8:
	.string	"imul"
.LC9:
	.string	"=="
.LC10:
	.string	"*"
.LC11:
	.string	"!="
.LC12:
	.string	"&&"
.LC13:
	.string	"ne"
	.align 8
.LC14:
	.string	"mov ebx, eax\npop eax\n%s eax, ebx\n"
.LC15:
	.string	"push eax\n"
	.align 8
.LC16:
	.string	"assignment requires a modifiable object\n"
	.align 8
.LC17:
	.string	"pop ebx\nmov dword ptr [ebx], eax\n"
.LC18:
	.string	"ge"
.LC19:
	.string	"l"
	.align 8
.LC20:
	.string	"pop ebx\ncmp ebx, eax\nmov eax, 0\nset%s al\n"
.LC21:
	.string	"||"
.LC22:
	.string	">="
.LC23:
	.string	"nz"
.LC24:
	.string	"?"
.LC25:
	.string	"e"
.LC26:
	.string	"cmp eax, 0\nj%s _%08d\n"
.LC27:
	.string	"+"
.LC28:
	.string	"z"
.LC29:
	.string	"-"
.LC30:
	.string	"="
.LC31:
	.string	"sub"
.LC32:
	.string	"add"
	.text
	.globl	expr
	.type	expr, @function
expr:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -132(%rbp)
	movq	$24, -8(%rbp)
.L357:
	cmpq	$89, -8(%rbp)
	ja	.L358
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L222(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L222(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L222:
	.long	.L302-.L222
	.long	.L301-.L222
	.long	.L300-.L222
	.long	.L299-.L222
	.long	.L359-.L222
	.long	.L297-.L222
	.long	.L296-.L222
	.long	.L358-.L222
	.long	.L295-.L222
	.long	.L294-.L222
	.long	.L358-.L222
	.long	.L293-.L222
	.long	.L292-.L222
	.long	.L291-.L222
	.long	.L290-.L222
	.long	.L289-.L222
	.long	.L358-.L222
	.long	.L288-.L222
	.long	.L358-.L222
	.long	.L287-.L222
	.long	.L286-.L222
	.long	.L285-.L222
	.long	.L284-.L222
	.long	.L283-.L222
	.long	.L282-.L222
	.long	.L281-.L222
	.long	.L280-.L222
	.long	.L279-.L222
	.long	.L278-.L222
	.long	.L277-.L222
	.long	.L276-.L222
	.long	.L358-.L222
	.long	.L275-.L222
	.long	.L274-.L222
	.long	.L273-.L222
	.long	.L272-.L222
	.long	.L271-.L222
	.long	.L270-.L222
	.long	.L269-.L222
	.long	.L268-.L222
	.long	.L267-.L222
	.long	.L266-.L222
	.long	.L265-.L222
	.long	.L264-.L222
	.long	.L263-.L222
	.long	.L262-.L222
	.long	.L358-.L222
	.long	.L358-.L222
	.long	.L261-.L222
	.long	.L358-.L222
	.long	.L260-.L222
	.long	.L259-.L222
	.long	.L258-.L222
	.long	.L257-.L222
	.long	.L256-.L222
	.long	.L255-.L222
	.long	.L254-.L222
	.long	.L358-.L222
	.long	.L253-.L222
	.long	.L252-.L222
	.long	.L251-.L222
	.long	.L250-.L222
	.long	.L249-.L222
	.long	.L248-.L222
	.long	.L247-.L222
	.long	.L246-.L222
	.long	.L245-.L222
	.long	.L244-.L222
	.long	.L243-.L222
	.long	.L242-.L222
	.long	.L241-.L222
	.long	.L359-.L222
	.long	.L239-.L222
	.long	.L238-.L222
	.long	.L237-.L222
	.long	.L236-.L222
	.long	.L235-.L222
	.long	.L234-.L222
	.long	.L233-.L222
	.long	.L232-.L222
	.long	.L231-.L222
	.long	.L230-.L222
	.long	.L229-.L222
	.long	.L228-.L222
	.long	.L227-.L222
	.long	.L226-.L222
	.long	.L225-.L222
	.long	.L224-.L222
	.long	.L223-.L222
	.long	.L221-.L222
	.text
.L260:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -114(%rbp)
	movq	$72, -8(%rbp)
	jmp	.L303
.L231:
	cmpb	$0, -106(%rbp)
	je	.L304
	movq	$77, -8(%rbp)
	jmp	.L303
.L304:
	movq	$23, -8(%rbp)
	jmp	.L303
.L281:
	cmpb	$0, -107(%rbp)
	je	.L306
	movq	$8, -8(%rbp)
	jmp	.L303
.L306:
	movq	$19, -8(%rbp)
	jmp	.L303
.L258:
	cmpb	$0, -111(%rbp)
	je	.L308
	movq	$30, -8(%rbp)
	jmp	.L303
.L308:
	movq	$62, -8(%rbp)
	jmp	.L303
.L276:
	leaq	.LC8(%rip), %rax
	movq	%rax, -40(%rbp)
	movq	$73, -8(%rbp)
	jmp	.L303
.L249:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -112(%rbp)
	movq	$89, -8(%rbp)
	jmp	.L303
.L290:
	cmpb	$0, -110(%rbp)
	je	.L311
	movq	$86, -8(%rbp)
	jmp	.L303
.L311:
	movq	$82, -8(%rbp)
	jmp	.L303
.L289:
	cmpb	$0, -108(%rbp)
	je	.L313
	movq	$43, -8(%rbp)
	jmp	.L303
.L313:
	movq	$88, -8(%rbp)
	jmp	.L303
.L229:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -111(%rbp)
	movq	$52, -8(%rbp)
	jmp	.L303
.L221:
	cmpb	$0, -112(%rbp)
	je	.L315
	movq	$28, -8(%rbp)
	jmp	.L303
.L315:
	movq	$35, -8(%rbp)
	jmp	.L303
.L254:
	cmpb	$0, -103(%rbp)
	je	.L317
	movq	$79, -8(%rbp)
	jmp	.L303
.L317:
	movq	$38, -8(%rbp)
	jmp	.L303
.L232:
	movl	$1, -92(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L303
.L292:
	cmpb	$0, -101(%rbp)
	je	.L319
	movq	$61, -8(%rbp)
	jmp	.L303
.L319:
	movq	$64, -8(%rbp)
	jmp	.L303
.L242:
	cmpl	$4, -132(%rbp)
	jne	.L321
	movq	$51, -8(%rbp)
	jmp	.L303
.L321:
	movq	$60, -8(%rbp)
	jmp	.L303
.L295:
	movl	$1, -96(%rbp)
	movq	$41, -8(%rbp)
	jmp	.L303
.L262:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -105(%rbp)
	movq	$58, -8(%rbp)
	jmp	.L303
.L256:
	movl	$0, -92(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L303
.L233:
	cmpb	$0, -113(%rbp)
	je	.L323
	movq	$11, -8(%rbp)
	jmp	.L303
.L323:
	movq	$50, -8(%rbp)
	jmp	.L303
.L301:
	movq	-24(%rbp), %rax
	movq	%rax, -72(%rbp)
	call	next
	movl	-132(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edi
	call	expr
	movq	$69, -8(%rbp)
	jmp	.L303
.L230:
	cmpl	$0, -84(%rbp)
	je	.L325
	movq	$83, -8(%rbp)
	jmp	.L303
.L325:
	movq	$34, -8(%rbp)
	jmp	.L303
.L283:
	movl	$0, -96(%rbp)
	movq	$41, -8(%rbp)
	jmp	.L303
.L234:
	movl	$1, -96(%rbp)
	movq	$41, -8(%rbp)
	jmp	.L303
.L241:
	movl	$1, %edi
	call	branch
	movq	$22, -8(%rbp)
	jmp	.L303
.L299:
	movq	-56(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L303
.L282:
	cmpl	$5, -132(%rbp)
	jne	.L327
	movq	$85, -8(%rbp)
	jmp	.L303
.L327:
	movq	$59, -8(%rbp)
	jmp	.L303
.L285:
	cmpb	$0, -99(%rbp)
	je	.L329
	movq	$6, -8(%rbp)
	jmp	.L303
.L329:
	movq	$36, -8(%rbp)
	jmp	.L303
.L271:
	cmpl	$1, -132(%rbp)
	jne	.L331
	movq	$74, -8(%rbp)
	jmp	.L303
.L331:
	movq	$22, -8(%rbp)
	jmp	.L303
.L235:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -104(%rbp)
	movq	$75, -8(%rbp)
	jmp	.L303
.L243:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -99(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L303
.L226:
	call	unary
	movq	$4, -8(%rbp)
	jmp	.L303
.L280:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -103(%rbp)
	movq	$56, -8(%rbp)
	jmp	.L303
.L293:
	leaq	.LC13(%rip), %rax
	movq	%rax, -56(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L303
.L294:
	movl	$1, -92(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L303
.L291:
	cmpb	$0, -98(%rbp)
	je	.L333
	movq	$70, -8(%rbp)
	jmp	.L303
.L333:
	movq	$22, -8(%rbp)
	jmp	.L303
.L248:
	movq	-64(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L303
.L259:
	movq	output(%rip), %rax
	movq	-72(%rbp), %rdx
	leaq	.LC14(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$32, -8(%rbp)
	jmp	.L303
.L287:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -106(%rbp)
	movq	$80, -8(%rbp)
	jmp	.L303
.L275:
	cmpl	$4, -132(%rbp)
	jne	.L335
	movq	$37, -8(%rbp)
	jmp	.L303
.L335:
	movq	$40, -8(%rbp)
	jmp	.L303
.L288:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$9, %edx
	movl	$1, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	needs_lvalue
	movl	-132(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edi
	call	expr
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$33, %edx
	movl	$1, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$71, -8(%rbp)
	jmp	.L303
.L267:
	cmpl	$3, -132(%rbp)
	jne	.L337
	movq	$45, -8(%rbp)
	jmp	.L303
.L337:
	movq	$39, -8(%rbp)
	jmp	.L303
.L244:
	leaq	.LC18(%rip), %rax
	movq	%rax, -64(%rbp)
	movq	$63, -8(%rbp)
	jmp	.L303
.L255:
	leaq	.LC19(%rip), %rax
	movq	%rax, -64(%rbp)
	movq	$63, -8(%rbp)
	jmp	.L303
.L251:
	movq	output(%rip), %rax
	movq	-72(%rbp), %rdx
	leaq	.LC20(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$32, -8(%rbp)
	jmp	.L303
.L252:
	movl	-132(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edi
	call	expr
	movq	$32, -8(%rbp)
	jmp	.L303
.L296:
	call	new_label
	movl	%eax, -76(%rbp)
	movl	-76(%rbp), %eax
	movl	%eax, -80(%rbp)
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -101(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L303
.L279:
	cmpb	$0, -109(%rbp)
	je	.L339
	movq	$20, -8(%rbp)
	jmp	.L303
.L339:
	movq	$0, -8(%rbp)
	jmp	.L303
.L269:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -102(%rbp)
	movq	$48, -8(%rbp)
	jmp	.L303
.L250:
	leaq	.LC23(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$65, -8(%rbp)
	jmp	.L303
.L224:
	movl	$1, -92(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L303
.L253:
	cmpb	$0, -105(%rbp)
	je	.L341
	movq	$42, -8(%rbp)
	jmp	.L303
.L341:
	movq	$76, -8(%rbp)
	jmp	.L303
.L227:
	movl	-92(%rbp), %eax
	movl	%eax, -88(%rbp)
	movq	$53, -8(%rbp)
	jmp	.L303
.L273:
	cmpl	$2, -132(%rbp)
	jne	.L343
	movq	$66, -8(%rbp)
	jmp	.L303
.L343:
	movq	$36, -8(%rbp)
	jmp	.L303
.L237:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -98(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L303
.L236:
	cmpb	$0, -104(%rbp)
	je	.L345
	movq	$9, -8(%rbp)
	jmp	.L303
.L345:
	movq	$26, -8(%rbp)
	jmp	.L303
.L261:
	cmpb	$0, -102(%rbp)
	je	.L347
	movq	$87, -8(%rbp)
	jmp	.L303
.L347:
	movq	$54, -8(%rbp)
	jmp	.L303
.L284:
	cmpl	$0, -132(%rbp)
	jne	.L349
	movq	$29, -8(%rbp)
	jmp	.L303
.L349:
	movq	$71, -8(%rbp)
	jmp	.L303
.L278:
	leaq	.LC25(%rip), %rax
	movq	%rax, -48(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L303
.L257:
	movl	-88(%rbp), %eax
	movl	%eax, -84(%rbp)
	movq	$81, -8(%rbp)
	jmp	.L303
.L246:
	movq	output(%rip), %rax
	movl	-80(%rbp), %ecx
	movq	-16(%rbp), %rdx
	leaq	.LC26(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	call	next
	movl	-132(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edi
	call	expr
	movq	output(%rip), %rax
	movl	-80(%rbp), %edx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$66, -8(%rbp)
	jmp	.L303
.L238:
	movq	-40(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L303
.L263:
	cmpb	$0, -97(%rbp)
	je	.L351
	movq	$17, -8(%rbp)
	jmp	.L303
.L351:
	movq	$71, -8(%rbp)
	jmp	.L303
.L297:
	cmpb	$0, -100(%rbp)
	je	.L353
	movq	$6, -8(%rbp)
	jmp	.L303
.L353:
	movq	$68, -8(%rbp)
	jmp	.L303
.L239:
	cmpb	$0, -114(%rbp)
	je	.L355
	movq	$55, -8(%rbp)
	jmp	.L303
.L355:
	movq	$67, -8(%rbp)
	jmp	.L303
.L274:
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L303
.L270:
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -108(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L303
.L247:
	leaq	.LC28(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$65, -8(%rbp)
	jmp	.L303
.L266:
	movl	-96(%rbp), %eax
	movl	%eax, -84(%rbp)
	movq	$81, -8(%rbp)
	jmp	.L303
.L265:
	movl	$1, -92(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L303
.L302:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -110(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L303
.L268:
	movl	$0, -88(%rbp)
	movq	$53, -8(%rbp)
	jmp	.L303
.L245:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -100(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L303
.L228:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$9, %edx
	movl	$1, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -109(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L303
.L223:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -107(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L303
.L272:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -113(%rbp)
	movq	$78, -8(%rbp)
	jmp	.L303
.L277:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -97(%rbp)
	movq	$44, -8(%rbp)
	jmp	.L303
.L264:
	movl	$1, -96(%rbp)
	movq	$41, -8(%rbp)
	jmp	.L303
.L225:
	leaq	.LC31(%rip), %rax
	movq	%rax, -32(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L303
.L300:
	movq	-48(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$73, -8(%rbp)
	jmp	.L303
.L286:
	leaq	.LC32(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L303
.L358:
	nop
.L303:
	jmp	.L357
.L359:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	expr, .-expr
	.section	.rodata
.LC33:
	.string	"jmp _%08d\n"
.LC34:
	.string	";"
.LC35:
	.string	"do"
.LC36:
	.string	"while"
.LC37:
	.string	"cmp eax, 0\nje _%08d\n"
	.text
	.globl	while_loop
	.type	while_loop, @function
while_loop:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$7, -8(%rbp)
.L379:
	cmpq	$11, -8(%rbp)
	ja	.L380
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L363(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L363(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L363:
	.long	.L372-.L363
	.long	.L371-.L363
	.long	.L370-.L363
	.long	.L369-.L363
	.long	.L380-.L363
	.long	.L368-.L363
	.long	.L381-.L363
	.long	.L366-.L363
	.long	.L365-.L363
	.long	.L364-.L363
	.long	.L380-.L363
	.long	.L362-.L363
	.text
.L365:
	movq	output(%rip), %rax
	movl	-28(%rbp), %edx
	leaq	.LC33(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	output(%rip), %rax
	movl	-24(%rbp), %edx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$6, -8(%rbp)
	jmp	.L373
.L371:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	match
	movq	$8, -8(%rbp)
	jmp	.L373
.L369:
	cmpb	$0, -30(%rbp)
	je	.L374
	movq	$1, -8(%rbp)
	jmp	.L373
.L374:
	movq	$11, -8(%rbp)
	jmp	.L373
.L362:
	call	line
	movq	$8, -8(%rbp)
	jmp	.L373
.L364:
	call	line
	movq	$0, -8(%rbp)
	jmp	.L373
.L368:
	call	new_label
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	emit_label
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -28(%rbp)
	call	new_label
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -24(%rbp)
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -29(%rbp)
	movzbl	-29(%rbp), %eax
	movb	%al, -30(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L373
.L372:
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	match
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	match
	movl	$0, %edi
	call	expr
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	match
	movq	output(%rip), %rax
	movl	-24(%rbp), %edx
	leaq	.LC37(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$3, -8(%rbp)
	jmp	.L373
.L366:
	movq	$5, -8(%rbp)
	jmp	.L373
.L370:
	cmpb	$0, -30(%rbp)
	je	.L377
	movq	$9, -8(%rbp)
	jmp	.L373
.L377:
	movq	$0, -8(%rbp)
	jmp	.L373
.L380:
	nop
.L373:
	jmp	.L379
.L381:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	while_loop, .-while_loop
	.globl	waiting_for
	.type	waiting_for, @function
waiting_for:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$6, -8(%rbp)
.L399:
	cmpq	$7, -8(%rbp)
	ja	.L401
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L385(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L385(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L385:
	.long	.L392-.L385
	.long	.L391-.L385
	.long	.L390-.L385
	.long	.L389-.L385
	.long	.L388-.L385
	.long	.L387-.L385
	.long	.L386-.L385
	.long	.L384-.L385
	.text
.L388:
	movl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L393
.L391:
	cmpl	$0, -16(%rbp)
	je	.L394
	movq	$2, -8(%rbp)
	jmp	.L393
.L394:
	movq	$4, -8(%rbp)
	jmp	.L393
.L389:
	cmpl	$0, -12(%rbp)
	setne	%al
	jmp	.L400
.L386:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -17(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L393
.L387:
	movq	input(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L393
.L392:
	cmpb	$0, -17(%rbp)
	je	.L397
	movq	$7, -8(%rbp)
	jmp	.L393
.L397:
	movq	$5, -8(%rbp)
	jmp	.L393
.L384:
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L393
.L390:
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L393
.L401:
	nop
.L393:
	jmp	.L399
.L400:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	waiting_for, .-waiting_for
	.globl	require
	.type	require, @function
require:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, %eax
	movq	%rsi, -32(%rbp)
	movb	%al, -20(%rbp)
	movq	$0, -8(%rbp)
.L410:
	cmpq	$2, -8(%rbp)
	je	.L403
	cmpq	$2, -8(%rbp)
	ja	.L412
	cmpq	$0, -8(%rbp)
	je	.L405
	cmpq	$1, -8(%rbp)
	jne	.L412
	jmp	.L411
.L405:
	movzbl	-20(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L407
	movq	$2, -8(%rbp)
	jmp	.L409
.L407:
	movq	$1, -8(%rbp)
	jmp	.L409
.L403:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	error
	movq	$1, -8(%rbp)
	jmp	.L409
.L412:
	nop
.L409:
	jmp	.L410
.L411:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	require, .-require
	.globl	try_match
	.type	try_match, @function
try_match:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$4, -8(%rbp)
.L425:
	cmpq	$4, -8(%rbp)
	ja	.L427
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L416(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L416(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L416:
	.long	.L420-.L416
	.long	.L419-.L416
	.long	.L418-.L416
	.long	.L417-.L416
	.long	.L415-.L416
	.text
.L415:
	movq	$2, -8(%rbp)
	jmp	.L421
.L419:
	call	next
	movq	$3, -8(%rbp)
	jmp	.L421
.L417:
	movzbl	-10(%rbp), %eax
	jmp	.L426
.L420:
	cmpb	$0, -10(%rbp)
	je	.L423
	movq	$1, -8(%rbp)
	jmp	.L421
.L423:
	movq	$3, -8(%rbp)
	jmp	.L421
.L418:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -9(%rbp)
	movzbl	-9(%rbp), %eax
	movb	%al, -10(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L421
.L427:
	nop
.L421:
	jmp	.L425
.L426:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	try_match, .-try_match
	.globl	eat_char
	.type	eat_char, @function
eat_char:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L434:
	cmpq	$2, -8(%rbp)
	je	.L429
	cmpq	$2, -8(%rbp)
	ja	.L436
	cmpq	$0, -8(%rbp)
	je	.L431
	cmpq	$1, -8(%rbp)
	jne	.L436
	jmp	.L435
.L431:
	movq	$2, -8(%rbp)
	jmp	.L433
.L429:
	movl	buflength(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	buflength(%rip), %eax
	addl	$1, %eax
	movl	%eax, buflength(%rip)
	movq	buffer(%rip), %rdx
	movl	-12(%rbp), %eax
	cltq
	addq	%rax, %rdx
	movzbl	curch(%rip), %eax
	movb	%al, (%rdx)
	call	next_char
	movq	$1, -8(%rbp)
	jmp	.L433
.L436:
	nop
.L433:
	jmp	.L434
.L435:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	eat_char, .-eat_char
	.section	.rodata
.LC38:
	.string	"%s:%d: error: "
	.text
	.globl	error
	.type	error, @function
error:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L443:
	cmpq	$2, -8(%rbp)
	je	.L438
	cmpq	$2, -8(%rbp)
	ja	.L445
	cmpq	$0, -8(%rbp)
	je	.L440
	cmpq	$1, -8(%rbp)
	jne	.L445
	jmp	.L444
.L440:
	movl	curln(%rip), %edx
	movq	inputname(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	buffer(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	errors(%rip), %eax
	addl	$1, %eax
	movl	%eax, errors(%rip)
	movq	$1, -8(%rbp)
	jmp	.L442
.L438:
	movq	$0, -8(%rbp)
	jmp	.L442
.L445:
	nop
.L442:
	jmp	.L443
.L444:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	error, .-error
	.globl	sym_lookup
	.type	sym_lookup, @function
sym_lookup:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$2, -8(%rbp)
.L461:
	cmpq	$8, -8(%rbp)
	ja	.L462
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L449(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L449(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L449:
	.long	.L454-.L449
	.long	.L462-.L449
	.long	.L453-.L449
	.long	.L452-.L449
	.long	.L462-.L449
	.long	.L451-.L449
	.long	.L462-.L449
	.long	.L450-.L449
	.long	.L448-.L449
	.text
.L448:
	movl	-20(%rbp), %eax
	subl	$1, %eax
	jmp	.L455
.L452:
	cmpl	$0, -16(%rbp)
	je	.L456
	movq	$0, -8(%rbp)
	jmp	.L458
.L456:
	movq	$8, -8(%rbp)
	jmp	.L458
.L451:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -20(%rbp)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L458
.L454:
	movl	-20(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L459
	movq	$5, -8(%rbp)
	jmp	.L458
.L459:
	movq	$7, -8(%rbp)
	jmp	.L458
.L450:
	movl	$-1, %eax
	jmp	.L455
.L453:
	movl	$0, -20(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L458
.L462:
	nop
.L458:
	jmp	.L461
.L455:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	sym_lookup, .-sym_lookup
	.section	.rodata
.LC39:
	.string	"cannot initialize a function\n"
	.align 8
.LC40:
	.string	"cannot initialize a parameter\n"
.LC41:
	.string	"{"
.LC42:
	.string	"%s: .quad %d\n"
.LC43:
	.string	"%s: .quad 0\n"
.LC44:
	.string	".section .text\n"
	.align 8
.LC45:
	.string	"a function implementation is illegal here\n"
	.align 8
.LC46:
	.string	"expected a constant expression, found '%s'\n"
.LC47:
	.string	","
.LC48:
	.string	"mov dword ptr [ebp%+d], eax\n"
.LC49:
	.string	".section .data\n"
	.text
	.globl	decl
	.type	decl, @function
decl:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -68(%rbp)
	movq	$3, -16(%rbp)
.L555:
	cmpq	$58, -16(%rbp)
	ja	.L556
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L466(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L466(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L466:
	.long	.L514-.L466
	.long	.L513-.L466
	.long	.L512-.L466
	.long	.L511-.L466
	.long	.L510-.L466
	.long	.L509-.L466
	.long	.L508-.L466
	.long	.L507-.L466
	.long	.L556-.L466
	.long	.L506-.L466
	.long	.L505-.L466
	.long	.L504-.L466
	.long	.L503-.L466
	.long	.L502-.L466
	.long	.L501-.L466
	.long	.L500-.L466
	.long	.L556-.L466
	.long	.L499-.L466
	.long	.L556-.L466
	.long	.L498-.L466
	.long	.L556-.L466
	.long	.L497-.L466
	.long	.L496-.L466
	.long	.L495-.L466
	.long	.L494-.L466
	.long	.L493-.L466
	.long	.L492-.L466
	.long	.L491-.L466
	.long	.L490-.L466
	.long	.L489-.L466
	.long	.L556-.L466
	.long	.L488-.L466
	.long	.L487-.L466
	.long	.L486-.L466
	.long	.L485-.L466
	.long	.L484-.L466
	.long	.L483-.L466
	.long	.L482-.L466
	.long	.L481-.L466
	.long	.L480-.L466
	.long	.L479-.L466
	.long	.L556-.L466
	.long	.L556-.L466
	.long	.L478-.L466
	.long	.L556-.L466
	.long	.L556-.L466
	.long	.L477-.L466
	.long	.L556-.L466
	.long	.L476-.L466
	.long	.L475-.L466
	.long	.L474-.L466
	.long	.L473-.L466
	.long	.L472-.L466
	.long	.L471-.L466
	.long	.L470-.L466
	.long	.L469-.L466
	.long	.L557-.L466
	.long	.L467-.L466
	.long	.L465-.L466
	.text
.L474:
	leaq	.LC39(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$38, -16(%rbp)
	jmp	.L515
.L493:
	cmpb	$0, -60(%rbp)
	je	.L516
	movq	$31, -16(%rbp)
	jmp	.L515
.L516:
	movq	$52, -16(%rbp)
	jmp	.L515
.L475:
	cmpb	$0, -59(%rbp)
	je	.L518
	movq	$46, -16(%rbp)
	jmp	.L515
.L518:
	movq	$1, -16(%rbp)
	jmp	.L515
.L472:
	movq	buffer(%rip), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	call	next
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -56(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L515
.L510:
	leaq	new_global(%rip), %rax
	movq	%rax, -32(%rbp)
	movq	$29, -16(%rbp)
	jmp	.L515
.L501:
	leaq	.LC40(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$38, -16(%rbp)
	jmp	.L515
.L500:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	new_local
	movl	%eax, -52(%rbp)
	movq	$55, -16(%rbp)
	jmp	.L515
.L488:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -60(%rbp)
	movq	$25, -16(%rbp)
	jmp	.L515
.L503:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	waiting_for
	movb	%al, -58(%rbp)
	movq	$26, -16(%rbp)
	jmp	.L515
.L470:
	movzbl	-61(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L521
	movq	$28, -16(%rbp)
	jmp	.L515
.L521:
	movq	$56, -16(%rbp)
	jmp	.L515
.L513:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	match
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	new_fn
	movb	$1, -62(%rbp)
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -57(%rbp)
	movq	$57, -16(%rbp)
	jmp	.L515
.L495:
	cmpb	$0, -54(%rbp)
	je	.L523
	movq	$39, -16(%rbp)
	jmp	.L515
.L523:
	movq	$17, -16(%rbp)
	jmp	.L515
.L511:
	movq	$43, -16(%rbp)
	jmp	.L515
.L494:
	movl	decl_module(%rip), %eax
	cmpl	%eax, -68(%rbp)
	jne	.L525
	movq	$2, -16(%rbp)
	jmp	.L515
.L525:
	movq	$36, -16(%rbp)
	jmp	.L515
.L497:
	movq	buffer(%rip), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -44(%rbp)
	movq	output(%rip), %rax
	movl	-44(%rbp), %ecx
	movq	-40(%rbp), %rdx
	leaq	.LC42(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$37, -16(%rbp)
	jmp	.L515
.L483:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -53(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L515
.L467:
	cmpb	$0, -57(%rbp)
	je	.L527
	movq	$6, -16(%rbp)
	jmp	.L515
.L527:
	movq	$55, -16(%rbp)
	jmp	.L515
.L492:
	cmpb	$0, -58(%rbp)
	je	.L529
	movq	$46, -16(%rbp)
	jmp	.L515
.L529:
	movq	$1, -16(%rbp)
	jmp	.L515
.L504:
	movl	decl_module(%rip), %eax
	cmpl	%eax, -68(%rbp)
	jne	.L531
	movq	$4, -16(%rbp)
	jmp	.L515
.L531:
	movq	$27, -16(%rbp)
	jmp	.L515
.L506:
	movl	$1, -48(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L515
.L502:
	cmpb	$0, -56(%rbp)
	je	.L533
	movq	$51, -16(%rbp)
	jmp	.L515
.L533:
	movq	$40, -16(%rbp)
	jmp	.L515
.L473:
	movl	decl_module(%rip), %eax
	cmpl	%eax, -68(%rbp)
	jne	.L535
	movq	$22, -16(%rbp)
	jmp	.L515
.L535:
	movq	$12, -16(%rbp)
	jmp	.L515
.L498:
	movq	output(%rip), %rax
	movq	-40(%rbp), %rdx
	leaq	.LC43(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$32, -16(%rbp)
	jmp	.L515
.L487:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$54, -16(%rbp)
	jmp	.L515
.L499:
	movzbl	-62(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L537
	movq	$19, -16(%rbp)
	jmp	.L515
.L537:
	movq	$32, -16(%rbp)
	jmp	.L515
.L479:
	movl	decl_local(%rip), %eax
	cmpl	%eax, -68(%rbp)
	jne	.L539
	movq	$15, -16(%rbp)
	jmp	.L515
.L539:
	movq	$11, -16(%rbp)
	jmp	.L515
.L469:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -55(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L515
.L508:
	movl	decl_module(%rip), %eax
	cmpl	%eax, -68(%rbp)
	sete	%al
	movzbl	%al, %eax
	leaq	.LC45(%rip), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	require
	movb	$1, -61(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	function
	movq	$55, -16(%rbp)
	jmp	.L515
.L491:
	leaq	new_param(%rip), %rax
	movq	%rax, -32(%rbp)
	movq	$29, -16(%rbp)
	jmp	.L515
.L481:
	movzbl	-62(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L541
	movq	$53, -16(%rbp)
	jmp	.L515
.L541:
	movq	$58, -16(%rbp)
	jmp	.L515
.L465:
	movl	$0, -48(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L515
.L485:
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	call	error
	movq	$37, -16(%rbp)
	jmp	.L515
.L476:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	match
	movq	$56, -16(%rbp)
	jmp	.L515
.L496:
	call	new_scope
	movq	$12, -16(%rbp)
	jmp	.L515
.L490:
	movl	decl_param(%rip), %eax
	cmpl	%eax, -68(%rbp)
	je	.L543
	movq	$48, -16(%rbp)
	jmp	.L515
.L543:
	movq	$56, -16(%rbp)
	jmp	.L515
.L471:
	movl	decl_param(%rip), %eax
	cmpl	%eax, -68(%rbp)
	je	.L545
	movq	$9, -16(%rbp)
	jmp	.L515
.L545:
	movq	$7, -16(%rbp)
	jmp	.L515
.L509:
	cmpb	$0, -53(%rbp)
	je	.L547
	movq	$35, -16(%rbp)
	jmp	.L515
.L547:
	movq	$54, -16(%rbp)
	jmp	.L515
.L486:
	cmpb	$0, -62(%rbp)
	je	.L549
	movq	$50, -16(%rbp)
	jmp	.L515
.L549:
	movq	$14, -16(%rbp)
	jmp	.L515
.L482:
	call	next
	movq	$32, -16(%rbp)
	jmp	.L515
.L505:
	cmpl	$0, -48(%rbp)
	setne	%al
	movzbl	%al, %eax
	movq	-24(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	require
	movq	$24, -16(%rbp)
	jmp	.L515
.L514:
	cmpb	$0, -55(%rbp)
	je	.L551
	movq	$33, -16(%rbp)
	jmp	.L515
.L551:
	movq	$24, -16(%rbp)
	jmp	.L515
.L477:
	movl	decl_param(%rip), %eax
	movl	%eax, %edi
	call	decl
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -59(%rbp)
	movq	$49, -16(%rbp)
	jmp	.L515
.L480:
	movl	token(%rip), %edx
	movl	token_int(%rip), %eax
	cmpl	%eax, %edx
	jne	.L553
	movq	$21, -16(%rbp)
	jmp	.L515
.L553:
	movq	$34, -16(%rbp)
	jmp	.L515
.L507:
	movl	$0, -48(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L515
.L484:
	movl	$0, %edi
	call	expr
	movq	offsets(%rip), %rdx
	movl	-52(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	output(%rip), %rax
	leaq	.LC48(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$54, -16(%rbp)
	jmp	.L515
.L489:
	movq	-40(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rax, %rdi
	call	*%rdx
	movq	$55, -16(%rbp)
	jmp	.L515
.L478:
	movb	$0, -62(%rbp)
	movb	$0, -61(%rbp)
	call	next
	movq	$31, -16(%rbp)
	jmp	.L515
.L512:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -54(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L515
.L556:
	nop
.L515:
	jmp	.L555
.L557:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	decl, .-decl
	.globl	sym_init
	.type	sym_init, @function
sym_init:
.LFB19:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movq	$0, -40(%rbp)
.L564:
	cmpq	$2, -40(%rbp)
	je	.L565
	cmpq	$2, -40(%rbp)
	ja	.L566
	cmpq	$0, -40(%rbp)
	je	.L561
	cmpq	$1, -40(%rbp)
	jne	.L566
	movl	ptr_size(%rip), %eax
	imull	-52(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, globals(%rip)
	movl	ptr_size(%rip), %eax
	movslq	%eax, %rdx
	movl	-52(%rbp), %eax
	cltq
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, is_fn(%rip)
	movl	ptr_size(%rip), %eax
	imull	-52(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, locals(%rip)
	movl	word_size(%rip), %eax
	movslq	%eax, %rdx
	movl	-52(%rbp), %eax
	cltq
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, offsets(%rip)
	movq	$2, -40(%rbp)
	jmp	.L562
.L561:
	movq	$1, -40(%rbp)
	jmp	.L562
.L566:
	nop
.L562:
	jmp	.L564
.L565:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	sym_init, .-sym_init
	.section	.rodata
.LC50:
	.string	"--"
.LC51:
	.string	"push eax\njmp _%08d\n"
.LC52:
	.string	"["
.LC53:
	.string	"call dword ptr [esp+%d]\n"
.LC54:
	.string	"add esp, %d\n"
.LC55:
	.string	"lea"
.LC56:
	.string	"mov"
.LC57:
	.string	"]"
.LC58:
	.string	"pop ebx\n%s eax, [eax*%d+ebx]\n"
.LC59:
	.string	"++"
.LC60:
	.string	"_%08d:\n"
	.text
	.globl	object
	.type	object, @function
object:
.LFB20:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	$4, -8(%rbp)
.L614:
	cmpq	$35, -8(%rbp)
	ja	.L615
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L570(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L570(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L570:
	.long	.L595-.L570
	.long	.L615-.L570
	.long	.L594-.L570
	.long	.L593-.L570
	.long	.L592-.L570
	.long	.L591-.L570
	.long	.L590-.L570
	.long	.L589-.L570
	.long	.L615-.L570
	.long	.L588-.L570
	.long	.L587-.L570
	.long	.L615-.L570
	.long	.L586-.L570
	.long	.L585-.L570
	.long	.L584-.L570
	.long	.L615-.L570
	.long	.L583-.L570
	.long	.L582-.L570
	.long	.L581-.L570
	.long	.L616-.L570
	.long	.L615-.L570
	.long	.L579-.L570
	.long	.L578-.L570
	.long	.L615-.L570
	.long	.L615-.L570
	.long	.L615-.L570
	.long	.L577-.L570
	.long	.L615-.L570
	.long	.L576-.L570
	.long	.L575-.L570
	.long	.L574-.L570
	.long	.L573-.L570
	.long	.L572-.L570
	.long	.L615-.L570
	.long	.L571-.L570
	.long	.L569-.L570
	.text
.L581:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -55(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L596
.L592:
	call	factor
	movq	$13, -8(%rbp)
	jmp	.L596
.L574:
	cmpb	$0, -57(%rbp)
	je	.L597
	movq	$12, -8(%rbp)
	jmp	.L596
.L597:
	movq	$29, -8(%rbp)
	jmp	.L596
.L584:
	cmpb	$0, -54(%rbp)
	je	.L599
	movq	$0, -8(%rbp)
	jmp	.L596
.L599:
	movq	$19, -8(%rbp)
	jmp	.L596
.L573:
	cmpb	$0, -59(%rbp)
	je	.L601
	movq	$3, -8(%rbp)
	jmp	.L596
.L601:
	movq	$2, -8(%rbp)
	jmp	.L596
.L586:
	movb	$1, lvalue(%rip)
	movq	$21, -8(%rbp)
	jmp	.L596
.L593:
	call	new_label
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %edi
	call	emit_label
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	$0, %edi
	call	expr
	movq	output(%rip), %rax
	movl	-40(%rbp), %edx
	leaq	.LC51(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	addl	$1, -52(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -40(%rbp)
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -59(%rbp)
	movq	$31, -8(%rbp)
	jmp	.L596
.L583:
	cmpb	$0, -56(%rbp)
	je	.L603
	movq	$35, -8(%rbp)
	jmp	.L596
.L603:
	movq	$18, -8(%rbp)
	jmp	.L596
.L579:
	movzbl	lvalue(%rip), %eax
	testb	%al, %al
	je	.L605
	movq	$22, -8(%rbp)
	jmp	.L596
.L605:
	movq	$5, -8(%rbp)
	jmp	.L596
.L577:
	cmpb	$0, -55(%rbp)
	je	.L607
	movq	$10, -8(%rbp)
	jmp	.L596
.L607:
	movq	$21, -8(%rbp)
	jmp	.L596
.L588:
	cmpb	$0, -53(%rbp)
	je	.L609
	movq	$28, -8(%rbp)
	jmp	.L596
.L609:
	movq	$17, -8(%rbp)
	jmp	.L596
.L585:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -53(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L596
.L572:
	call	new_label
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -48(%rbp)
	call	new_label
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, -40(%rbp)
	movq	output(%rip), %rax
	movl	-48(%rbp), %edx
	leaq	.LC33(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$3, -8(%rbp)
	jmp	.L596
.L582:
	leaq	.LC52(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -54(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L596
.L590:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	match
	movl	word_size(%rip), %eax
	imull	-52(%rbp), %eax
	movl	%eax, %edx
	movq	output(%rip), %rax
	leaq	.LC53(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	-52(%rbp), %eax
	leal	1(%rax), %edx
	movl	word_size(%rip), %eax
	imull	%eax, %edx
	movq	output(%rip), %rax
	leaq	.LC54(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$13, -8(%rbp)
	jmp	.L596
.L571:
	cmpb	$0, -58(%rbp)
	je	.L612
	movq	$32, -8(%rbp)
	jmp	.L596
.L612:
	movq	$6, -8(%rbp)
	jmp	.L596
.L578:
	leaq	.LC55(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L596
.L576:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$9, %edx
	movl	$1, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$0, -52(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	waiting_for
	movb	%al, -58(%rbp)
	movq	$34, -8(%rbp)
	jmp	.L596
.L591:
	leaq	.LC56(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L596
.L587:
	movb	$1, lvalue(%rip)
	movq	$21, -8(%rbp)
	jmp	.L596
.L595:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$9, %edx
	movl	$1, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$0, %edi
	call	expr
	leaq	.LC57(%rip), %rax
	movq	%rax, %rdi
	call	match
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -57(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L596
.L589:
	movl	word_size(%rip), %ecx
	movq	output(%rip), %rax
	movq	-16(%rbp), %rdx
	leaq	.LC58(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$13, -8(%rbp)
	jmp	.L596
.L569:
	movb	$1, lvalue(%rip)
	movq	$21, -8(%rbp)
	jmp	.L596
.L575:
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -56(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L596
.L594:
	movq	output(%rip), %rax
	movl	-48(%rbp), %edx
	leaq	.LC60(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	output(%rip), %rax
	movl	-40(%rbp), %edx
	leaq	.LC33(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	output(%rip), %rax
	movl	-44(%rbp), %edx
	leaq	.LC60(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$6, -8(%rbp)
	jmp	.L596
.L615:
	nop
.L596:
	jmp	.L614
.L616:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE20:
	.size	object, .-object
	.globl	emit_label
	.type	emit_label, @function
emit_label:
.LFB23:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L622:
	cmpq	$0, -8(%rbp)
	je	.L618
	cmpq	$1, -8(%rbp)
	jne	.L624
	movq	output(%rip), %rax
	movl	-20(%rbp), %edx
	leaq	.LC60(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$0, -8(%rbp)
	jmp	.L620
.L618:
	movl	-20(%rbp), %eax
	jmp	.L623
.L624:
	nop
.L620:
	jmp	.L622
.L623:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE23:
	.size	emit_label, .-emit_label
	.globl	needs_lvalue
	.type	needs_lvalue, @function
needs_lvalue:
.LFB24:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L634:
	cmpq	$3, -8(%rbp)
	je	.L626
	cmpq	$3, -8(%rbp)
	ja	.L636
	cmpq	$2, -8(%rbp)
	je	.L628
	cmpq	$2, -8(%rbp)
	ja	.L636
	cmpq	$0, -8(%rbp)
	je	.L629
	cmpq	$1, -8(%rbp)
	jne	.L636
	jmp	.L635
.L626:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	error
	movq	$0, -8(%rbp)
	jmp	.L631
.L629:
	movb	$0, lvalue(%rip)
	movq	$1, -8(%rbp)
	jmp	.L631
.L628:
	movzbl	lvalue(%rip), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L632
	movq	$3, -8(%rbp)
	jmp	.L631
.L632:
	movq	$0, -8(%rbp)
	jmp	.L631
.L636:
	nop
.L631:
	jmp	.L634
.L635:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE24:
	.size	needs_lvalue, .-needs_lvalue
	.section	.rodata
.LC61:
	.string	".intel_syntax noprefix\n"
	.text
	.globl	program
	.type	program, @function
program:
.LFB25:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L650:
	cmpq	$5, -8(%rbp)
	ja	.L651
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L640(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L640(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L640:
	.long	.L645-.L640
	.long	.L644-.L640
	.long	.L643-.L640
	.long	.L652-.L640
	.long	.L641-.L640
	.long	.L639-.L640
	.text
.L641:
	movq	$1, -8(%rbp)
	jmp	.L646
.L644:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$23, %edx
	movl	$1, %esi
	leaq	.LC61(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$0, errors(%rip)
	movq	$0, -8(%rbp)
	jmp	.L646
.L639:
	cmpl	$0, -12(%rbp)
	je	.L648
	movq	$3, -8(%rbp)
	jmp	.L646
.L648:
	movq	$2, -8(%rbp)
	jmp	.L646
.L645:
	movq	input(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L646
.L643:
	movl	decl_module(%rip), %eax
	movl	%eax, %edi
	call	decl
	movq	$0, -8(%rbp)
	jmp	.L646
.L651:
	nop
.L646:
	jmp	.L650
.L652:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE25:
	.size	program, .-program
	.globl	new_global
	.type	new_global, @function
new_global:
.LFB26:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L659:
	cmpq	$2, -8(%rbp)
	je	.L660
	cmpq	$2, -8(%rbp)
	ja	.L661
	cmpq	$0, -8(%rbp)
	je	.L656
	cmpq	$1, -8(%rbp)
	jne	.L661
	movl	global_no(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	global_no(%rip), %eax
	addl	$1, %eax
	movl	%eax, global_no(%rip)
	movq	globals(%rip), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rax, (%rdx)
	movq	$2, -8(%rbp)
	jmp	.L657
.L656:
	movq	$1, -8(%rbp)
	jmp	.L657
.L661:
	nop
.L657:
	jmp	.L659
.L660:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE26:
	.size	new_global, .-new_global
	.section	.rodata
	.align 8
.LC62:
	.string	"mov ebx, eax\nmov eax, [ebx]\n%s dword ptr [ebx], 1\n"
	.align 8
.LC63:
	.string	"assignment operator '%s' requires a modifiable object\n"
.LC64:
	.string	"!"
	.align 8
.LC65:
	.string	"cmp eax, 0\nmov eax, 0\nsete al\n"
.LC66:
	.string	"neg eax\n"
	.text
	.globl	unary
	.type	unary, @function
unary:
.LFB27:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$17, -8(%rbp)
.L693:
	cmpq	$19, -8(%rbp)
	ja	.L694
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L665(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L665(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L665:
	.long	.L680-.L665
	.long	.L694-.L665
	.long	.L695-.L665
	.long	.L678-.L665
	.long	.L694-.L665
	.long	.L677-.L665
	.long	.L676-.L665
	.long	.L694-.L665
	.long	.L675-.L665
	.long	.L674-.L665
	.long	.L673-.L665
	.long	.L672-.L665
	.long	.L671-.L665
	.long	.L670-.L665
	.long	.L669-.L665
	.long	.L668-.L665
	.long	.L667-.L665
	.long	.L666-.L665
	.long	.L694-.L665
	.long	.L664-.L665
	.text
.L669:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -19(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L681
.L668:
	cmpb	$0, -20(%rbp)
	je	.L682
	movq	$10, -8(%rbp)
	jmp	.L681
.L682:
	movq	$14, -8(%rbp)
	jmp	.L681
.L671:
	cmpb	$0, -18(%rbp)
	je	.L684
	movq	$5, -8(%rbp)
	jmp	.L681
.L684:
	movq	$9, -8(%rbp)
	jmp	.L681
.L675:
	cmpb	$0, -17(%rbp)
	je	.L686
	movq	$6, -8(%rbp)
	jmp	.L681
.L686:
	movq	$3, -8(%rbp)
	jmp	.L681
.L678:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -18(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L681
.L667:
	movq	output(%rip), %rax
	movq	-16(%rbp), %rdx
	leaq	.LC62(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	leaq	.LC63(%rip), %rax
	movq	%rax, %rdi
	call	needs_lvalue
	call	next
	movq	$2, -8(%rbp)
	jmp	.L681
.L672:
	leaq	.LC31(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L681
.L674:
	call	object
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -20(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L681
.L670:
	cmpb	$0, -19(%rbp)
	je	.L688
	movq	$10, -8(%rbp)
	jmp	.L681
.L688:
	movq	$2, -8(%rbp)
	jmp	.L681
.L664:
	leaq	.LC32(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L681
.L666:
	leaq	.LC64(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -17(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L681
.L676:
	call	unary
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$30, %edx
	movl	$1, %esi
	leaq	.LC65(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$2, -8(%rbp)
	jmp	.L681
.L677:
	call	unary
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$8, %edx
	movl	$1, %esi
	leaq	.LC66(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$2, -8(%rbp)
	jmp	.L681
.L673:
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -21(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L681
.L680:
	cmpb	$0, -21(%rbp)
	je	.L690
	movq	$19, -8(%rbp)
	jmp	.L681
.L690:
	movq	$11, -8(%rbp)
	jmp	.L681
.L694:
	nop
.L681:
	jmp	.L693
.L695:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE27:
	.size	unary, .-unary
	.section	.rodata
.LC67:
	.string	"}"
.LC68:
	.string	"int"
.LC69:
	.string	"char"
.LC70:
	.string	"return"
.LC71:
	.string	"bool"
	.text
	.globl	line
	.type	line, @function
line:
.LFB28:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$3, -8(%rbp)
.L752:
	cmpq	$33, -8(%rbp)
	ja	.L753
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L699(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L699(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L699:
	.long	.L729-.L699
	.long	.L728-.L699
	.long	.L727-.L699
	.long	.L726-.L699
	.long	.L725-.L699
	.long	.L724-.L699
	.long	.L723-.L699
	.long	.L753-.L699
	.long	.L722-.L699
	.long	.L721-.L699
	.long	.L720-.L699
	.long	.L719-.L699
	.long	.L718-.L699
	.long	.L717-.L699
	.long	.L716-.L699
	.long	.L715-.L699
	.long	.L754-.L699
	.long	.L713-.L699
	.long	.L712-.L699
	.long	.L711-.L699
	.long	.L753-.L699
	.long	.L710-.L699
	.long	.L709-.L699
	.long	.L753-.L699
	.long	.L708-.L699
	.long	.L707-.L699
	.long	.L706-.L699
	.long	.L705-.L699
	.long	.L704-.L699
	.long	.L703-.L699
	.long	.L702-.L699
	.long	.L701-.L699
	.long	.L700-.L699
	.long	.L698-.L699
	.text
.L712:
	movl	return_to(%rip), %edx
	movq	output(%rip), %rax
	leaq	.LC33(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$11, -8(%rbp)
	jmp	.L730
.L707:
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -12(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L730
.L725:
	leaq	.LC67(%rip), %rax
	movq	%rax, %rdi
	call	waiting_for
	movb	%al, -19(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L730
.L702:
	leaq	.LC68(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -15(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L730
.L716:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -11(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L730
.L715:
	call	if_branch
	movq	$16, -8(%rbp)
	jmp	.L730
.L701:
	movl	decl_local(%rip), %eax
	movl	%eax, %edi
	call	decl
	movq	$16, -8(%rbp)
	jmp	.L730
.L718:
	leaq	.LC67(%rip), %rax
	movq	%rax, %rdi
	call	match
	movq	$16, -8(%rbp)
	jmp	.L730
.L722:
	call	while_loop
	movq	$16, -8(%rbp)
	jmp	.L730
.L728:
	cmpb	$0, -19(%rbp)
	je	.L731
	movq	$32, -8(%rbp)
	jmp	.L730
.L731:
	movq	$12, -8(%rbp)
	jmp	.L730
.L726:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -10(%rbp)
	movq	$28, -8(%rbp)
	jmp	.L730
.L708:
	movl	$0, %edi
	call	expr
	movq	$33, -8(%rbp)
	jmp	.L730
.L710:
	cmpb	$0, -13(%rbp)
	je	.L734
	movq	$6, -8(%rbp)
	jmp	.L730
.L734:
	movq	$13, -8(%rbp)
	jmp	.L730
.L706:
	leaq	.LC69(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -14(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L730
.L719:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	match
	movq	$16, -8(%rbp)
	jmp	.L730
.L721:
	cmpb	$0, -11(%rbp)
	je	.L736
	movq	$8, -8(%rbp)
	jmp	.L730
.L736:
	movq	$30, -8(%rbp)
	jmp	.L730
.L717:
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L730
.L711:
	cmpb	$0, -17(%rbp)
	je	.L738
	movq	$24, -8(%rbp)
	jmp	.L730
.L738:
	movq	$33, -8(%rbp)
	jmp	.L730
.L700:
	call	line
	movq	$4, -8(%rbp)
	jmp	.L730
.L713:
	movl	decl_local(%rip), %eax
	movl	%eax, %edi
	call	decl
	movq	$16, -8(%rbp)
	jmp	.L730
.L723:
	movl	decl_local(%rip), %eax
	movl	%eax, %edi
	call	decl
	movq	$16, -8(%rbp)
	jmp	.L730
.L705:
	cmpb	$0, -12(%rbp)
	je	.L740
	movq	$29, -8(%rbp)
	jmp	.L730
.L740:
	movq	$14, -8(%rbp)
	jmp	.L730
.L709:
	cmpb	$0, -14(%rbp)
	je	.L742
	movq	$31, -8(%rbp)
	jmp	.L730
.L742:
	movq	$2, -8(%rbp)
	jmp	.L730
.L704:
	cmpb	$0, -10(%rbp)
	je	.L744
	movq	$15, -8(%rbp)
	jmp	.L730
.L744:
	movq	$25, -8(%rbp)
	jmp	.L730
.L724:
	cmpb	$0, -15(%rbp)
	je	.L746
	movq	$17, -8(%rbp)
	jmp	.L730
.L746:
	movq	$26, -8(%rbp)
	jmp	.L730
.L698:
	cmpb	$0, -18(%rbp)
	je	.L748
	movq	$18, -8(%rbp)
	jmp	.L730
.L748:
	movq	$11, -8(%rbp)
	jmp	.L730
.L720:
	leaq	.LC70(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -9(%rbp)
	movzbl	-9(%rbp), %eax
	movb	%al, -18(%rbp)
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	waiting_for
	movb	%al, -17(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L730
.L729:
	cmpb	$0, -16(%rbp)
	je	.L750
	movq	$4, -8(%rbp)
	jmp	.L730
.L750:
	movq	$10, -8(%rbp)
	jmp	.L730
.L703:
	call	while_loop
	movq	$16, -8(%rbp)
	jmp	.L730
.L727:
	leaq	.LC71(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -13(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L730
.L753:
	nop
.L730:
	jmp	.L752
.L754:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE28:
	.size	line, .-line
	.globl	new_fn
	.type	new_fn, @function
new_fn:
.LFB29:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$24, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L761:
	cmpq	$2, -8(%rbp)
	je	.L756
	cmpq	$2, -8(%rbp)
	ja	.L763
	cmpq	$0, -8(%rbp)
	je	.L758
	cmpq	$1, -8(%rbp)
	jne	.L763
	jmp	.L762
.L758:
	movq	is_fn(%rip), %rdx
	movl	global_no(%rip), %eax
	cltq
	addq	%rdx, %rax
	movb	$1, (%rax)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	new_global
	movq	$1, -8(%rbp)
	jmp	.L760
.L756:
	movq	$0, -8(%rbp)
	jmp	.L760
.L763:
	nop
.L760:
	jmp	.L761
.L762:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE29:
	.size	new_fn, .-new_fn
	.section	.rodata
.LC72:
	.string	"r"
	.text
	.globl	lex_init
	.type	lex_init, @function
lex_init:
.LFB30:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$0, -16(%rbp)
.L770:
	cmpq	$2, -16(%rbp)
	je	.L771
	cmpq	$2, -16(%rbp)
	ja	.L772
	cmpq	$0, -16(%rbp)
	je	.L767
	cmpq	$1, -16(%rbp)
	jne	.L772
	movq	-24(%rbp), %rax
	movq	%rax, inputname(%rip)
	movq	-24(%rbp), %rax
	leaq	.LC72(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, input(%rip)
	movl	$1, curln(%rip)
	movl	-28(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, buffer(%rip)
	call	next_char
	call	next
	movq	$2, -16(%rbp)
	jmp	.L768
.L767:
	movq	$1, -16(%rbp)
	jmp	.L768
.L772:
	nop
.L768:
	jmp	.L770
.L771:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE30:
	.size	lex_init, .-lex_init
	.globl	new_scope
	.type	new_scope, @function
new_scope:
.LFB31:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L779:
	cmpq	$2, -8(%rbp)
	je	.L780
	cmpq	$2, -8(%rbp)
	ja	.L781
	cmpq	$0, -8(%rbp)
	je	.L776
	cmpq	$1, -8(%rbp)
	jne	.L781
	movq	$0, -8(%rbp)
	jmp	.L777
.L776:
	movl	$0, local_no(%rip)
	movl	$0, param_no(%rip)
	movq	$2, -8(%rbp)
	jmp	.L777
.L781:
	nop
.L777:
	jmp	.L779
.L780:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE31:
	.size	new_scope, .-new_scope
	.section	.rodata
.LC73:
	.string	"mov eax, %s\n"
.LC74:
	.string	"false"
.LC75:
	.string	".ascii %s\n"
.LC76:
	.string	"%s eax, [ebp%+d]\n"
.LC77:
	.string	"%s eax, [%s]\n"
.LC78:
	.string	"true"
.LC79:
	.string	".byte 0\n.section .text\n"
.LC80:
	.string	"mov eax, offset _%08d\n"
	.align 8
.LC81:
	.string	"expected an expression, found '%s'\n"
.LC82:
	.string	"mov eax, %d\n"
.LC83:
	.string	".section .rodata\n"
.LC84:
	.string	"no symbol '%s' declared\n"
	.text
	.globl	factor
	.type	factor, @function
factor:
.LFB32:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	$29, -8(%rbp)
.L877:
	cmpq	$63, -8(%rbp)
	ja	.L878
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L785(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L785(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L785:
	.long	.L836-.L785
	.long	.L835-.L785
	.long	.L878-.L785
	.long	.L879-.L785
	.long	.L833-.L785
	.long	.L832-.L785
	.long	.L831-.L785
	.long	.L830-.L785
	.long	.L829-.L785
	.long	.L828-.L785
	.long	.L827-.L785
	.long	.L826-.L785
	.long	.L825-.L785
	.long	.L824-.L785
	.long	.L823-.L785
	.long	.L878-.L785
	.long	.L822-.L785
	.long	.L878-.L785
	.long	.L878-.L785
	.long	.L821-.L785
	.long	.L820-.L785
	.long	.L878-.L785
	.long	.L878-.L785
	.long	.L819-.L785
	.long	.L818-.L785
	.long	.L817-.L785
	.long	.L878-.L785
	.long	.L816-.L785
	.long	.L815-.L785
	.long	.L814-.L785
	.long	.L813-.L785
	.long	.L812-.L785
	.long	.L878-.L785
	.long	.L811-.L785
	.long	.L810-.L785
	.long	.L809-.L785
	.long	.L808-.L785
	.long	.L807-.L785
	.long	.L806-.L785
	.long	.L805-.L785
	.long	.L804-.L785
	.long	.L803-.L785
	.long	.L802-.L785
	.long	.L801-.L785
	.long	.L800-.L785
	.long	.L799-.L785
	.long	.L798-.L785
	.long	.L878-.L785
	.long	.L797-.L785
	.long	.L878-.L785
	.long	.L796-.L785
	.long	.L795-.L785
	.long	.L794-.L785
	.long	.L878-.L785
	.long	.L793-.L785
	.long	.L792-.L785
	.long	.L791-.L785
	.long	.L790-.L785
	.long	.L789-.L785
	.long	.L788-.L785
	.long	.L787-.L785
	.long	.L878-.L785
	.long	.L786-.L785
	.long	.L784-.L785
	.text
.L796:
	cmpb	$0, -66(%rbp)
	je	.L837
	movq	$38, -8(%rbp)
	jmp	.L839
.L837:
	movq	$11, -8(%rbp)
	jmp	.L839
.L817:
	movb	$1, lvalue(%rip)
	movq	$63, -8(%rbp)
	jmp	.L839
.L794:
	cmpl	$0, -52(%rbp)
	js	.L840
	movq	$58, -8(%rbp)
	jmp	.L839
.L840:
	movq	$3, -8(%rbp)
	jmp	.L839
.L833:
	cmpb	$0, -67(%rbp)
	je	.L842
	movq	$35, -8(%rbp)
	jmp	.L839
.L842:
	movq	$40, -8(%rbp)
	jmp	.L839
.L813:
	leaq	.LC55(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L839
.L786:
	movq	buffer(%rip), %rdx
	movq	output(%rip), %rax
	leaq	.LC73(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	call	next
	movq	$3, -8(%rbp)
	jmp	.L839
.L823:
	cmpb	$0, -62(%rbp)
	je	.L844
	movq	$36, -8(%rbp)
	jmp	.L839
.L844:
	movq	$31, -8(%rbp)
	jmp	.L839
.L791:
	movq	buffer(%rip), %rdx
	movq	output(%rip), %rax
	leaq	.LC73(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	call	next
	movq	$3, -8(%rbp)
	jmp	.L839
.L812:
	leaq	.LC74(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -61(%rbp)
	movq	$55, -8(%rbp)
	jmp	.L839
.L825:
	leaq	.LC55(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L839
.L829:
	movq	buffer(%rip), %rdx
	movq	output(%rip), %rax
	leaq	.LC75(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	call	next
	movq	$33, -8(%rbp)
	jmp	.L839
.L799:
	movl	$1, -48(%rbp)
	movq	$43, -8(%rbp)
	jmp	.L839
.L793:
	movq	is_fn(%rip), %rdx
	movl	-56(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L846
	movq	$12, -8(%rbp)
	jmp	.L839
.L846:
	movq	$19, -8(%rbp)
	jmp	.L839
.L835:
	movq	offsets(%rip), %rdx
	movl	-52(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movq	output(%rip), %rax
	movq	-16(%rbp), %rdx
	leaq	.LC76(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$3, -8(%rbp)
	jmp	.L839
.L819:
	movq	globals(%rip), %rdx
	movl	-56(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movq	output(%rip), %rax
	movq	-24(%rbp), %rdx
	leaq	.LC77(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$3, -8(%rbp)
	jmp	.L839
.L822:
	leaq	.LC56(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L839
.L818:
	movl	$1, -48(%rbp)
	movq	$43, -8(%rbp)
	jmp	.L839
.L808:
	leaq	.LC78(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -67(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L839
.L790:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -64(%rbp)
	movq	$46, -8(%rbp)
	jmp	.L839
.L826:
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -65(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L839
.L828:
	movq	buffer(%rip), %rdx
	movl	global_no(%rip), %ecx
	movq	globals(%rip), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	sym_lookup
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, -56(%rbp)
	movq	buffer(%rip), %rdx
	movl	local_no(%rip), %ecx
	movq	locals(%rip), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	sym_lookup
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -52(%rbp)
	movq	$37, -8(%rbp)
	jmp	.L839
.L824:
	movb	$1, lvalue(%rip)
	movq	$63, -8(%rbp)
	jmp	.L839
.L784:
	cmpl	$0, -56(%rbp)
	js	.L849
	movq	$54, -8(%rbp)
	jmp	.L839
.L849:
	movq	$52, -8(%rbp)
	jmp	.L839
.L795:
	movl	token(%rip), %edx
	movl	token_str(%rip), %eax
	cmpl	%eax, %edx
	jne	.L851
	movq	$41, -8(%rbp)
	jmp	.L839
.L851:
	movq	$0, -8(%rbp)
	jmp	.L839
.L821:
	movzbl	lvalue(%rip), %eax
	testb	%al, %al
	je	.L853
	movq	$39, -8(%rbp)
	jmp	.L839
.L853:
	movq	$48, -8(%rbp)
	jmp	.L839
.L804:
	movl	$0, -60(%rbp)
	movq	$44, -8(%rbp)
	jmp	.L839
.L792:
	cmpb	$0, -61(%rbp)
	je	.L855
	movq	$36, -8(%rbp)
	jmp	.L839
.L855:
	movq	$5, -8(%rbp)
	jmp	.L839
.L787:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$23, %edx
	movl	$1, %esi
	leaq	.LC79(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	output(%rip), %rax
	movl	-44(%rbp), %edx
	leaq	.LC80(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$3, -8(%rbp)
	jmp	.L839
.L788:
	movl	token(%rip), %edx
	movl	token_char(%rip), %eax
	cmpl	%eax, %edx
	jne	.L857
	movq	$56, -8(%rbp)
	jmp	.L839
.L857:
	movq	$51, -8(%rbp)
	jmp	.L839
.L831:
	cmpb	$0, -65(%rbp)
	je	.L859
	movq	$25, -8(%rbp)
	jmp	.L839
.L859:
	movq	$57, -8(%rbp)
	jmp	.L839
.L816:
	cmpl	$0, -52(%rbp)
	js	.L861
	movq	$45, -8(%rbp)
	jmp	.L839
.L861:
	movq	$28, -8(%rbp)
	jmp	.L839
.L806:
	movb	$1, lvalue(%rip)
	movq	$63, -8(%rbp)
	jmp	.L839
.L789:
	movzbl	lvalue(%rip), %eax
	testb	%al, %al
	je	.L863
	movq	$30, -8(%rbp)
	jmp	.L839
.L863:
	movq	$16, -8(%rbp)
	jmp	.L839
.L810:
	leaq	.LC81(%rip), %rax
	movq	%rax, %rdi
	call	error
	movq	$3, -8(%rbp)
	jmp	.L839
.L797:
	leaq	.LC56(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L839
.L815:
	movl	$0, -48(%rbp)
	movq	$43, -8(%rbp)
	jmp	.L839
.L800:
	movq	output(%rip), %rax
	movl	-60(%rbp), %edx
	leaq	.LC82(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	call	next
	movq	$3, -8(%rbp)
	jmp	.L839
.L832:
	movl	token(%rip), %edx
	movl	token_ident(%rip), %eax
	cmpl	%eax, %edx
	jne	.L865
	movq	$9, -8(%rbp)
	jmp	.L839
.L865:
	movq	$7, -8(%rbp)
	jmp	.L839
.L811:
	movl	token(%rip), %edx
	movl	token_str(%rip), %eax
	cmpl	%eax, %edx
	jne	.L867
	movq	$8, -8(%rbp)
	jmp	.L839
.L867:
	movq	$60, -8(%rbp)
	jmp	.L839
.L807:
	cmpl	$0, -56(%rbp)
	js	.L869
	movq	$24, -8(%rbp)
	jmp	.L839
.L869:
	movq	$27, -8(%rbp)
	jmp	.L839
.L803:
	movq	output(%rip), %rax
	movq	%rax, %rcx
	movl	$17, %edx
	movl	$1, %esi
	leaq	.LC83(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	call	new_label
	movl	%eax, -40(%rbp)
	movl	-40(%rbp), %eax
	movl	%eax, %edi
	call	emit_label
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -44(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L839
.L827:
	movl	$0, %edi
	call	expr
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	match
	movq	$3, -8(%rbp)
	jmp	.L839
.L802:
	cmpb	$0, -63(%rbp)
	je	.L871
	movq	$10, -8(%rbp)
	jmp	.L839
.L871:
	movq	$34, -8(%rbp)
	jmp	.L839
.L836:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -63(%rbp)
	movq	$42, -8(%rbp)
	jmp	.L839
.L798:
	cmpb	$0, -64(%rbp)
	je	.L873
	movq	$13, -8(%rbp)
	jmp	.L839
.L873:
	movq	$63, -8(%rbp)
	jmp	.L839
.L805:
	leaq	.LC55(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L839
.L830:
	movl	token(%rip), %edx
	movl	token_int(%rip), %eax
	cmpl	%eax, %edx
	jne	.L875
	movq	$62, -8(%rbp)
	jmp	.L839
.L875:
	movq	$59, -8(%rbp)
	jmp	.L839
.L809:
	movl	$1, -60(%rbp)
	movq	$44, -8(%rbp)
	jmp	.L839
.L814:
	movq	$20, -8(%rbp)
	jmp	.L839
.L801:
	cmpl	$0, -48(%rbp)
	setne	%al
	movzbl	%al, %eax
	leaq	.LC84(%rip), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	require
	call	next
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -66(%rbp)
	movq	$50, -8(%rbp)
	jmp	.L839
.L820:
	movb	$0, lvalue(%rip)
	leaq	.LC78(%rip), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -62(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L839
.L878:
	nop
.L839:
	jmp	.L877
.L879:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE32:
	.size	factor, .-factor
	.globl	next_char
	.type	next_char, @function
next_char:
.LFB33:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L890:
	cmpq	$4, -8(%rbp)
	je	.L881
	cmpq	$4, -8(%rbp)
	ja	.L892
	cmpq	$2, -8(%rbp)
	je	.L883
	cmpq	$2, -8(%rbp)
	ja	.L892
	cmpq	$0, -8(%rbp)
	je	.L884
	cmpq	$1, -8(%rbp)
	je	.L885
	jmp	.L892
.L881:
	movzbl	curch(%rip), %eax
	jmp	.L891
.L885:
	movq	input(%rip), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movb	%al, curch(%rip)
	movq	$4, -8(%rbp)
	jmp	.L887
.L884:
	movzbl	curch(%rip), %eax
	cmpb	$10, %al
	jne	.L888
	movq	$2, -8(%rbp)
	jmp	.L887
.L888:
	movq	$1, -8(%rbp)
	jmp	.L887
.L883:
	movl	curln(%rip), %eax
	addl	$1, %eax
	movl	%eax, curln(%rip)
	movq	$1, -8(%rbp)
	jmp	.L887
.L892:
	nop
.L887:
	jmp	.L890
.L891:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE33:
	.size	next_char, .-next_char
	.globl	prev_char
	.type	prev_char, @function
prev_char:
.LFB34:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$0, -8(%rbp)
.L899:
	cmpq	$2, -8(%rbp)
	je	.L894
	cmpq	$2, -8(%rbp)
	ja	.L901
	cmpq	$0, -8(%rbp)
	je	.L896
	cmpq	$1, -8(%rbp)
	jne	.L901
	movl	$0, %eax
	jmp	.L900
.L896:
	movq	$2, -8(%rbp)
	jmp	.L898
.L894:
	movq	input(%rip), %rdx
	movzbl	curch(%rip), %eax
	movsbl	%al, %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	ungetc@PLT
	movzbl	-20(%rbp), %eax
	movb	%al, curch(%rip)
	movq	$1, -8(%rbp)
	jmp	.L898
.L901:
	nop
.L898:
	jmp	.L899
.L900:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE34:
	.size	prev_char, .-prev_char
	.section	.rodata
.LC85:
	.string	"else"
.LC86:
	.string	":"
	.text
	.globl	branch
	.type	branch, @function
branch:
.LFB36:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, %eax
	movb	%al, -36(%rbp)
	movq	$9, -8(%rbp)
.L926:
	cmpq	$13, -8(%rbp)
	ja	.L927
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L905(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L905(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L905:
	.long	.L928-.L905
	.long	.L916-.L905
	.long	.L915-.L905
	.long	.L914-.L905
	.long	.L913-.L905
	.long	.L912-.L905
	.long	.L911-.L905
	.long	.L910-.L905
	.long	.L909-.L905
	.long	.L908-.L905
	.long	.L907-.L905
	.long	.L927-.L905
	.long	.L906-.L905
	.long	.L904-.L905
	.text
.L913:
	movl	$1, %edi
	call	expr
	movq	$6, -8(%rbp)
	jmp	.L918
.L906:
	call	new_label
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -24(%rbp)
	call	new_label
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	output(%rip), %rax
	movl	-24(%rbp), %edx
	leaq	.LC37(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$1, -8(%rbp)
	jmp	.L918
.L909:
	call	line
	movq	$2, -8(%rbp)
	jmp	.L918
.L916:
	cmpb	$0, -36(%rbp)
	je	.L919
	movq	$4, -8(%rbp)
	jmp	.L918
.L919:
	movq	$5, -8(%rbp)
	jmp	.L918
.L914:
	leaq	.LC85(%rip), %rax
	movq	%rax, %rdi
	call	try_match
	movb	%al, -25(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L918
.L908:
	movq	$12, -8(%rbp)
	jmp	.L918
.L904:
	cmpb	$0, -25(%rbp)
	je	.L921
	movq	$8, -8(%rbp)
	jmp	.L918
.L921:
	movq	$2, -8(%rbp)
	jmp	.L918
.L911:
	movq	output(%rip), %rax
	movl	-20(%rbp), %edx
	leaq	.LC33(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	output(%rip), %rax
	movl	-24(%rbp), %edx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$7, -8(%rbp)
	jmp	.L918
.L912:
	call	line
	movq	$6, -8(%rbp)
	jmp	.L918
.L907:
	leaq	.LC86(%rip), %rax
	movq	%rax, %rdi
	call	match
	movl	$1, %edi
	call	expr
	movq	$2, -8(%rbp)
	jmp	.L918
.L910:
	cmpb	$0, -36(%rbp)
	je	.L924
	movq	$10, -8(%rbp)
	jmp	.L918
.L924:
	movq	$3, -8(%rbp)
	jmp	.L918
.L915:
	movq	output(%rip), %rax
	movl	-20(%rbp), %edx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$0, -8(%rbp)
	jmp	.L918
.L927:
	nop
.L918:
	jmp	.L926
.L928:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE36:
	.size	branch, .-branch
	.section	.rodata
	.align 8
.LC87:
	.string	"%s:%d: error: expected '%s', found '%s'\n"
	.text
	.globl	match
	.type	match, @function
match:
.LFB37:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$3, -8(%rbp)
.L941:
	cmpq	$5, -8(%rbp)
	ja	.L942
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L932(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L932(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L932:
	.long	.L936-.L932
	.long	.L935-.L932
	.long	.L942-.L932
	.long	.L934-.L932
	.long	.L933-.L932
	.long	.L943-.L932
	.text
.L933:
	movq	buffer(%rip), %rsi
	movl	curln(%rip), %edx
	movq	inputname(%rip), %rax
	movq	-24(%rbp), %rcx
	movq	%rsi, %r8
	movq	%rax, %rsi
	leaq	.LC87(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	errors(%rip), %eax
	addl	$1, %eax
	movl	%eax, errors(%rip)
	movq	$1, -8(%rbp)
	jmp	.L937
.L935:
	call	next
	movq	$5, -8(%rbp)
	jmp	.L937
.L934:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	see
	movb	%al, -9(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L937
.L936:
	cmpb	$0, -9(%rbp)
	je	.L939
	movq	$1, -8(%rbp)
	jmp	.L937
.L939:
	movq	$4, -8(%rbp)
	jmp	.L937
.L942:
	nop
.L937:
	jmp	.L941
.L943:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE37:
	.size	match, .-match
	.section	.rodata
.LC88:
	.string	"Usage: cc <file>"
.LC89:
	.string	"w"
.LC90:
	.string	"a.s"
	.align 8
.LC91:
	.string	"malloc"
	.string	"calloc"
	.string	"free"
	.string	"atoi"
	.string	"fopen"
	.string	"fclose"
	.string	"fgetc"
	.string	"ungetc"
	.string	"feof"
	.string	"fputs"
	.string	"fprintf"
	.string	"puts"
	.string	"printf"
	.string	"isalpha"
	.string	"isdigit"
	.string	"isalnum"
	.string	"strlen"
	.string	"strcmp"
	.string	"strchr"
	.string	"strcpy"
	.string	"strdup"
	.string	"\377\377\377\377"
	.text
	.globl	main
	.type	main, @function
main:
.LFB38:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	$3, decl_param(%rip)
	nop
.L945:
	movl	$2, decl_local(%rip)
	nop
.L946:
	movl	$1, decl_module(%rip)
	nop
.L947:
	movb	$0, lvalue(%rip)
	nop
.L948:
	movl	$0, return_to(%rip)
	nop
.L949:
	movl	$0, label_no(%rip)
	nop
.L950:
	movq	$0, offsets(%rip)
	nop
.L951:
	movl	$0, param_no(%rip)
	nop
.L952:
	movl	$0, local_no(%rip)
	nop
.L953:
	movq	$0, locals(%rip)
	nop
.L954:
	movq	$0, is_fn(%rip)
	nop
.L955:
	movl	$0, global_no(%rip)
	nop
.L956:
	movq	$0, globals(%rip)
	nop
.L957:
	movl	$0, errors(%rip)
	nop
.L958:
	movl	$4, token_str(%rip)
	nop
.L959:
	movl	$3, token_char(%rip)
	nop
.L960:
	movl	$2, token_int(%rip)
	nop
.L961:
	movl	$1, token_ident(%rip)
	nop
.L962:
	movl	$0, token_other(%rip)
	nop
.L963:
	movl	$0, token(%rip)
	nop
.L964:
	movl	$0, buflength(%rip)
	nop
.L965:
	movq	$0, buffer(%rip)
	nop
.L966:
	movb	$0, curch(%rip)
	nop
.L967:
	movl	$0, curln(%rip)
	nop
.L968:
	movq	$0, input(%rip)
	nop
.L969:
	movq	$0, inputname(%rip)
	nop
.L970:
	movq	$0, output(%rip)
	nop
.L971:
	movl	$4, word_size(%rip)
	nop
.L972:
	movl	$4, ptr_size(%rip)
	nop
.L973:
	movq	$0, _TIG_IZ_BYwQ_envp(%rip)
	nop
.L974:
	movq	$0, _TIG_IZ_BYwQ_argv(%rip)
	nop
.L975:
	movl	$0, _TIG_IZ_BYwQ_argc(%rip)
	nop
	nop
.L976:
.L977:
#APP
# 416 "Fedjmike_mini-c_cc.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-BYwQ--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_BYwQ_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_BYwQ_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_BYwQ_envp(%rip)
	nop
	movq	$0, -24(%rbp)
.L994:
	cmpq	$11, -24(%rbp)
	ja	.L995
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L980(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L980(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L980:
	.long	.L987-.L980
	.long	.L995-.L980
	.long	.L986-.L980
	.long	.L995-.L980
	.long	.L985-.L980
	.long	.L984-.L980
	.long	.L983-.L980
	.long	.L995-.L980
	.long	.L982-.L980
	.long	.L981-.L980
	.long	.L995-.L980
	.long	.L979-.L980
	.text
.L985:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	new_fn
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	addq	$1, %rax
	addq	%rax, -32(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L988
.L982:
	leaq	.LC88(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -24(%rbp)
	jmp	.L988
.L979:
	leaq	.LC89(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC90(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, output(%rip)
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movl	$256, %esi
	movq	%rax, %rdi
	call	lex_init
	movl	$256, %edi
	call	sym_init
	leaq	.LC91(%rip), %rax
	movq	%rax, -32(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L988
.L981:
	movl	$1, %eax
	jmp	.L989
.L983:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$-1, %al
	je	.L990
	movq	$4, -24(%rbp)
	jmp	.L988
.L990:
	movq	$5, -24(%rbp)
	jmp	.L988
.L984:
	call	program
	movq	$2, -24(%rbp)
	jmp	.L988
.L987:
	cmpl	$2, -36(%rbp)
	je	.L992
	movq	$8, -24(%rbp)
	jmp	.L988
.L992:
	movq	$11, -24(%rbp)
	jmp	.L988
.L986:
	movl	errors(%rip), %eax
	testl	%eax, %eax
	setne	%al
	movzbl	%al, %eax
	jmp	.L989
.L995:
	nop
.L988:
	jmp	.L994
.L989:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE38:
	.size	main, .-main
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
