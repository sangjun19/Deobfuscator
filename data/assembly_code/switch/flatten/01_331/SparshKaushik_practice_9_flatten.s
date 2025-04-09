	.file	"SparshKaushik_practice_9_flatten.c"
	.text
	.globl	head
	.bss
	.align 8
	.type	head, @object
	.size	head, 8
head:
	.zero	8
	.globl	_TIG_IZ_dpZn_envp
	.align 8
	.type	_TIG_IZ_dpZn_envp, @object
	.size	_TIG_IZ_dpZn_envp, 8
_TIG_IZ_dpZn_envp:
	.zero	8
	.globl	_TIG_IZ_dpZn_argv
	.align 8
	.type	_TIG_IZ_dpZn_argv, @object
	.size	_TIG_IZ_dpZn_argv, 8
_TIG_IZ_dpZn_argv:
	.zero	8
	.globl	_TIG_IZ_dpZn_argc
	.align 4
	.type	_TIG_IZ_dpZn_argc, @object
	.size	_TIG_IZ_dpZn_argc, 4
_TIG_IZ_dpZn_argc:
	.zero	4
	.globl	tail
	.align 8
	.type	tail, @object
	.size	tail, 8
tail:
	.zero	8
	.section	.rodata
.LC0:
	.string	"List is empty"
	.text
	.globl	delete
	.type	delete, @function
delete:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -16(%rbp)
.L16:
	cmpq	$7, -16(%rbp)
	ja	.L17
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
	.long	.L9-.L4
	.long	.L17-.L4
	.long	.L17-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L18-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movq	head(%rip), %rax
	testq	%rax, %rax
	jne	.L10
	movq	$7, -16(%rbp)
	jmp	.L12
.L10:
	movq	$6, -16(%rbp)
	jmp	.L12
.L8:
	movq	head(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	head(%rip), %rax
	movq	24(%rax), %rax
	movq	%rax, head(%rip)
	movq	head(%rip), %rax
	movq	$0, 32(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -16(%rbp)
	jmp	.L12
.L5:
	movq	head(%rip), %rax
	movq	%rax, %rdx
	movq	tail(%rip), %rax
	cmpq	%rax, %rdx
	jne	.L13
	movq	$0, -16(%rbp)
	jmp	.L12
.L13:
	movq	$3, -16(%rbp)
	jmp	.L12
.L9:
	movq	head(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, head(%rip)
	movq	$0, tail(%rip)
	movq	$5, -16(%rbp)
	jmp	.L12
.L3:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L12
.L17:
	nop
.L12:
	jmp	.L16
.L18:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	delete, .-delete
	.section	.rodata
	.align 8
.LC1:
	.string	"1. Insert\n2. Delete\n3. Display\n4. Exit\nEnter your choice: "
.LC2:
	.string	"%d"
.LC3:
	.string	"Enter the id: "
.LC4:
	.string	"Enter the name: "
.LC5:
	.string	"%s"
.LC6:
	.string	"Invalid choice"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, tail(%rip)
	nop
.L20:
	movq	$0, head(%rip)
	nop
.L21:
	movq	$0, _TIG_IZ_dpZn_envp(%rip)
	nop
.L22:
	movq	$0, _TIG_IZ_dpZn_argv(%rip)
	nop
.L23:
	movl	$0, _TIG_IZ_dpZn_argc(%rip)
	nop
	nop
.L24:
.L25:
#APP
# 140 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-dpZn--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_dpZn_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_dpZn_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_dpZn_envp(%rip)
	nop
	movq	$12, -40(%rbp)
.L47:
	cmpq	$14, -40(%rbp)
	ja	.L50
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L36-.L28
	.long	.L35-.L28
	.long	.L34-.L28
	.long	.L33-.L28
	.long	.L50-.L28
	.long	.L50-.L28
	.long	.L32-.L28
	.long	.L50-.L28
	.long	.L50-.L28
	.long	.L31-.L28
	.long	.L50-.L28
	.long	.L50-.L28
	.long	.L30-.L28
	.long	.L29-.L28
	.long	.L27-.L28
	.text
.L27:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L48
	jmp	.L49
.L30:
	movq	$9, -40(%rbp)
	jmp	.L38
.L35:
	movl	-48(%rbp), %eax
	cmpl	$4, %eax
	je	.L39
	cmpl	$4, %eax
	jg	.L40
	cmpl	$3, %eax
	je	.L41
	cmpl	$3, %eax
	jg	.L40
	cmpl	$1, %eax
	je	.L42
	cmpl	$2, %eax
	je	.L43
	jmp	.L40
.L39:
	movq	$9, -40(%rbp)
	jmp	.L44
.L41:
	movq	$0, -40(%rbp)
	jmp	.L44
.L43:
	movq	$3, -40(%rbp)
	jmp	.L44
.L42:
	movq	$6, -40(%rbp)
	jmp	.L44
.L40:
	movq	$2, -40(%rbp)
	nop
.L44:
	jmp	.L38
.L33:
	call	delete
	movq	$9, -40(%rbp)
	jmp	.L38
.L31:
	movl	-48(%rbp), %eax
	cmpl	$4, %eax
	je	.L45
	movq	$13, -40(%rbp)
	jmp	.L38
.L45:
	movq	$14, -40(%rbp)
	jmp	.L38
.L29:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -40(%rbp)
	jmp	.L38
.L32:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-44(%rbp), %eax
	leaq	-32(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	insert
	movq	$9, -40(%rbp)
	jmp	.L38
.L36:
	call	display
	movq	$9, -40(%rbp)
	jmp	.L38
.L34:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -40(%rbp)
	jmp	.L38
.L50:
	nop
.L38:
	jmp	.L47
.L49:
	call	__stack_chk_fail@PLT
.L48:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	insert
	.type	insert, @function
insert:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$5, -16(%rbp)
.L64:
	cmpq	$7, -16(%rbp)
	ja	.L65
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L54(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L54(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L54:
	.long	.L59-.L54
	.long	.L65-.L54
	.long	.L58-.L54
	.long	.L65-.L54
	.long	.L57-.L54
	.long	.L56-.L54
	.long	.L55-.L54
	.long	.L66-.L54
	.text
.L57:
	movq	head(%rip), %rax
	testq	%rax, %rax
	jne	.L60
	movq	$6, -16(%rbp)
	jmp	.L62
.L60:
	movq	$2, -16(%rbp)
	jmp	.L62
.L55:
	movq	-24(%rbp), %rax
	movq	%rax, head(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, tail(%rip)
	movq	$7, -16(%rbp)
	jmp	.L62
.L56:
	movq	$0, -16(%rbp)
	jmp	.L62
.L59:
	movl	$40, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	leaq	4(%rax), %rdx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	movq	-24(%rbp), %rax
	movq	$0, 24(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 32(%rax)
	movq	$4, -16(%rbp)
	jmp	.L62
.L58:
	movq	head(%rip), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 32(%rax)
	movq	head(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 24(%rax)
	movq	-24(%rbp), %rax
	movq	%rax, head(%rip)
	movq	$7, -16(%rbp)
	jmp	.L62
.L65:
	nop
.L62:
	jmp	.L64
.L66:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	insert, .-insert
	.section	.rodata
.LC7:
	.string	"College #%d \n"
	.text
	.globl	display
	.type	display, @function
display:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L77:
	cmpq	$6, -8(%rbp)
	je	.L67
	cmpq	$6, -8(%rbp)
	ja	.L78
	cmpq	$3, -8(%rbp)
	je	.L70
	cmpq	$3, -8(%rbp)
	ja	.L78
	cmpq	$1, -8(%rbp)
	je	.L71
	cmpq	$2, -8(%rbp)
	je	.L72
	jmp	.L78
.L71:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	addq	$4, %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rax, -16(%rbp)
	movl	$10, %edi
	call	putchar@PLT
	movq	$3, -8(%rbp)
	jmp	.L73
.L70:
	cmpq	$0, -16(%rbp)
	je	.L74
	movq	$1, -8(%rbp)
	jmp	.L73
.L74:
	movq	$6, -8(%rbp)
	jmp	.L73
.L72:
	movq	head(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L73
.L78:
	nop
.L73:
	jmp	.L77
.L67:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	display, .-display
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
