	.file	"timber-they_AdventOfCode15_8_flatten.c"
	.text
	.globl	_TIG_IZ_tHtG_argc
	.bss
	.align 4
	.type	_TIG_IZ_tHtG_argc, @object
	.size	_TIG_IZ_tHtG_argc, 4
_TIG_IZ_tHtG_argc:
	.zero	4
	.globl	_TIG_IZ_tHtG_argv
	.align 8
	.type	_TIG_IZ_tHtG_argv, @object
	.size	_TIG_IZ_tHtG_argv, 8
_TIG_IZ_tHtG_argv:
	.zero	8
	.globl	_TIG_IZ_tHtG_envp
	.align 8
	.type	_TIG_IZ_tHtG_envp, @object
	.size	_TIG_IZ_tHtG_envp, 8
_TIG_IZ_tHtG_envp:
	.zero	8
	.text
	.globl	encodedSize
	.type	encodedSize, @function
encodedSize:
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
	movq	$4, -16(%rbp)
.L20:
	cmpq	$13, -16(%rbp)
	ja	.L22
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
	.long	.L22-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L22-.L4
	.long	.L22-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L22-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L10:
	movq	$8, -16(%rbp)
	jmp	.L14
.L5:
	movl	-28(%rbp), %eax
	jmp	.L21
.L8:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, -28(%rbp)
	addl	$2, -28(%rbp)
	movq	-40(%rbp), %rax
	subq	$1, %rax
	movq	%rax, -24(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L14
.L13:
	cmpq	$0, -24(%rbp)
	je	.L16
	movq	$11, -16(%rbp)
	jmp	.L14
.L16:
	movq	$12, -16(%rbp)
	jmp	.L14
.L11:
	addl	$1, -28(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L14
.L6:
	addl	$1, -28(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L14
.L7:
	movq	-40(%rbp), %rax
	subq	$1, %rax
	movq	%rax, -24(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L14
.L3:
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movl	$34, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -24(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L14
.L9:
	cmpq	$0, -24(%rbp)
	je	.L18
	movq	$3, -16(%rbp)
	jmp	.L14
.L18:
	movq	$9, -16(%rbp)
	jmp	.L14
.L12:
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movl	$92, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L14
.L22:
	nop
.L14:
	jmp	.L20
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	encodedSize, .-encodedSize
	.globl	part2
	.type	part2, @function
part2:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -24(%rbp)
.L40:
	cmpq	$10, -24(%rbp)
	ja	.L43
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L26(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L26(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L26:
	.long	.L33-.L26
	.long	.L32-.L26
	.long	.L31-.L26
	.long	.L43-.L26
	.long	.L30-.L26
	.long	.L29-.L26
	.long	.L43-.L26
	.long	.L28-.L26
	.long	.L27-.L26
	.long	.L43-.L26
	.long	.L25-.L26
	.text
.L30:
	movq	-72(%rbp), %rdx
	leaq	-48(%rbp), %rcx
	leaq	-40(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	getline@PLT
	movq	%rax, -32(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L34
.L27:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -60(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	encodedSize
	movl	%eax, -56(%rbp)
	movl	-56(%rbp), %eax
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	subl	-60(%rbp), %eax
	addl	%eax, -64(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L34
.L32:
	cmpq	$0, -32(%rbp)
	jle	.L35
	movq	$7, -24(%rbp)
	jmp	.L34
.L35:
	movq	$10, -24(%rbp)
	jmp	.L34
.L29:
	movl	-64(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L41
	jmp	.L42
.L25:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -24(%rbp)
	jmp	.L34
.L33:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movl	$0, -64(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L34
.L28:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	je	.L38
	movq	$8, -24(%rbp)
	jmp	.L34
.L38:
	movq	$10, -24(%rbp)
	jmp	.L34
.L31:
	movq	$0, -24(%rbp)
	jmp	.L34
.L43:
	nop
.L34:
	jmp	.L40
.L42:
	call	__stack_chk_fail@PLT
.L41:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	part2, .-part2
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"in8"
.LC2:
	.string	"Part1: %d\n"
.LC3:
	.string	"Part2: %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_tHtG_envp(%rip)
	nop
.L45:
	movq	$0, _TIG_IZ_tHtG_argv(%rip)
	nop
.L46:
	movl	$0, _TIG_IZ_tHtG_argc(%rip)
	nop
	nop
.L47:
.L48:
#APP
# 108 "timber-they_AdventOfCode15_8.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-tHtG--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_tHtG_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_tHtG_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_tHtG_envp(%rip)
	nop
	movq	$1, -24(%rbp)
.L54:
	cmpq	$2, -24(%rbp)
	je	.L49
	cmpq	$2, -24(%rbp)
	ja	.L56
	cmpq	$0, -24(%rbp)
	je	.L51
	cmpq	$1, -24(%rbp)
	jne	.L56
	movq	$0, -24(%rbp)
	jmp	.L52
.L51:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	part1
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	part2
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$2, -24(%rbp)
	jmp	.L52
.L49:
	movl	$0, %eax
	jmp	.L55
.L56:
	nop
.L52:
	jmp	.L54
.L55:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.globl	part1
	.type	part1, @function
part1:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -24(%rbp)
.L74:
	cmpq	$11, -24(%rbp)
	ja	.L77
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L60(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L60(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L60:
	.long	.L67-.L60
	.long	.L66-.L60
	.long	.L65-.L60
	.long	.L64-.L60
	.long	.L63-.L60
	.long	.L62-.L60
	.long	.L77-.L60
	.long	.L77-.L60
	.long	.L61-.L60
	.long	.L77-.L60
	.long	.L77-.L60
	.long	.L59-.L60
	.text
.L63:
	movl	-64(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L75
	jmp	.L76
.L61:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$4, -24(%rbp)
	jmp	.L69
.L66:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -60(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	memorySize
	movl	%eax, -56(%rbp)
	movl	-56(%rbp), %eax
	movl	%eax, -52(%rbp)
	movl	-60(%rbp), %eax
	subl	-52(%rbp), %eax
	addl	%eax, -64(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L69
.L64:
	movq	$0, -24(%rbp)
	jmp	.L69
.L59:
	cmpq	$0, -32(%rbp)
	jle	.L70
	movq	$5, -24(%rbp)
	jmp	.L69
.L70:
	movq	$8, -24(%rbp)
	jmp	.L69
.L62:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	je	.L72
	movq	$1, -24(%rbp)
	jmp	.L69
.L72:
	movq	$8, -24(%rbp)
	jmp	.L69
.L67:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movl	$0, -64(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L69
.L65:
	movq	-72(%rbp), %rdx
	leaq	-48(%rbp), %rcx
	leaq	-40(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	getline@PLT
	movq	%rax, -32(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L69
.L77:
	nop
.L69:
	jmp	.L74
.L76:
	call	__stack_chk_fail@PLT
.L75:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	part1, .-part1
	.section	.rodata
	.align 8
.LC4:
	.string	"Unexpected escape: %c (for %s)\n"
	.text
	.globl	memorySize
	.type	memorySize, @function
memorySize:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$4, -16(%rbp)
.L100:
	cmpq	$15, -16(%rbp)
	ja	.L102
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L81(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L81(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L81:
	.long	.L90-.L81
	.long	.L89-.L81
	.long	.L102-.L81
	.long	.L102-.L81
	.long	.L88-.L81
	.long	.L87-.L81
	.long	.L102-.L81
	.long	.L86-.L81
	.long	.L85-.L81
	.long	.L102-.L81
	.long	.L84-.L81
	.long	.L102-.L81
	.long	.L83-.L81
	.long	.L82-.L81
	.long	.L102-.L81
	.long	.L80-.L81
	.text
.L88:
	movq	$0, -16(%rbp)
	jmp	.L91
.L80:
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %edx
	movq	stderr(%rip), %rax
	movq	-40(%rbp), %rcx
	leaq	.LC4(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L83:
	movq	-40(%rbp), %rax
	addq	$2, %rax
	movl	$92, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -40(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L91
.L85:
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$120, %eax
	je	.L92
	cmpl	$120, %eax
	jg	.L93
	cmpl	$34, %eax
	je	.L94
	cmpl	$92, %eax
	je	.L95
	jmp	.L93
.L92:
	movq	$10, -16(%rbp)
	jmp	.L96
.L94:
	movq	$1, -16(%rbp)
	jmp	.L96
.L95:
	movq	$7, -16(%rbp)
	jmp	.L96
.L93:
	movq	$15, -16(%rbp)
	nop
.L96:
	jmp	.L91
.L89:
	subl	$1, -20(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L91
.L82:
	cmpq	$0, -40(%rbp)
	je	.L97
	movq	$8, -16(%rbp)
	jmp	.L91
.L97:
	movq	$5, -16(%rbp)
	jmp	.L91
.L87:
	movl	-20(%rbp), %eax
	jmp	.L101
.L84:
	subl	$3, -20(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L91
.L90:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, -20(%rbp)
	subl	$2, -20(%rbp)
	subq	$2, -40(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L91
.L86:
	subl	$1, -20(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L91
.L102:
	nop
.L91:
	jmp	.L100
.L101:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	memorySize, .-memorySize
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
