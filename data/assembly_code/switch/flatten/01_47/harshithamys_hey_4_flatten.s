	.file	"harshithamys_hey_4_flatten.c"
	.text
	.globl	front
	.bss
	.align 4
	.type	front, @object
	.size	front, 4
front:
	.zero	4
	.globl	_TIG_IZ_93cQ_envp
	.align 8
	.type	_TIG_IZ_93cQ_envp, @object
	.size	_TIG_IZ_93cQ_envp, 8
_TIG_IZ_93cQ_envp:
	.zero	8
	.globl	rear
	.align 4
	.type	rear, @object
	.size	rear, 4
rear:
	.zero	4
	.globl	_TIG_IZ_93cQ_argc
	.align 4
	.type	_TIG_IZ_93cQ_argc, @object
	.size	_TIG_IZ_93cQ_argc, 4
_TIG_IZ_93cQ_argc:
	.zero	4
	.globl	pjob
	.align 32
	.type	pjob, @object
	.size	pjob, 40
pjob:
	.zero	40
	.globl	_TIG_IZ_93cQ_argv
	.align 8
	.type	_TIG_IZ_93cQ_argv, @object
	.size	_TIG_IZ_93cQ_argv, 8
_TIG_IZ_93cQ_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Info\t PR"
.LC1:
	.string	"%d\t %d\n"
.LC2:
	.string	"Queue is Empty"
	.text
	.globl	display
	.type	display, @function
display:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$6, -8(%rbp)
.L16:
	cmpq	$8, -8(%rbp)
	ja	.L17
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
	.long	.L17-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L17-.L4
	.long	.L3-.L4
	.text
.L7:
	movl	rear(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L10
	movq	$8, -8(%rbp)
	jmp	.L12
.L10:
	movq	$1, -8(%rbp)
	jmp	.L12
.L3:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	4+pjob(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rcx
	leaq	pjob(%rip), %rax
	movl	(%rcx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L12
.L8:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L12
.L5:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L14
	movq	$3, -8(%rbp)
	jmp	.L12
.L14:
	movq	$5, -8(%rbp)
	jmp	.L12
.L6:
	movl	front(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
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
.LFE0:
	.size	display, .-display
	.section	.rodata
.LC3:
	.string	"Overflow"
	.align 8
.LC4:
	.string	"Enter information and its priority: "
.LC5:
	.string	"%d %d"
	.text
	.globl	insert
	.type	insert, @function
insert:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -16(%rbp)
.L36:
	cmpq	$10, -16(%rbp)
	ja	.L39
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L29-.L22
	.long	.L28-.L22
	.long	.L27-.L22
	.long	.L39-.L22
	.long	.L39-.L22
	.long	.L26-.L22
	.long	.L40-.L22
	.long	.L39-.L22
	.long	.L24-.L22
	.long	.L23-.L22
	.long	.L21-.L22
	.text
.L24:
	movl	rear(%rip), %eax
	addl	$1, %eax
	movl	%eax, rear(%rip)
	movq	$10, -16(%rbp)
	jmp	.L30
.L28:
	movl	rear(%rip), %eax
	cmpl	$4, %eax
	jne	.L31
	movq	$5, -16(%rbp)
	jmp	.L30
.L31:
	movq	$0, -16(%rbp)
	jmp	.L30
.L23:
	movl	rear(%rip), %eax
	cmpl	$-1, %eax
	jne	.L33
	movq	$2, -16(%rbp)
	jmp	.L30
.L33:
	movq	$8, -16(%rbp)
	jmp	.L30
.L26:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -16(%rbp)
	jmp	.L30
.L21:
	movl	rear(%rip), %edx
	movl	-24(%rbp), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	leaq	pjob(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movl	rear(%rip), %edx
	movl	-20(%rbp), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	leaq	4+pjob(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$6, -16(%rbp)
	jmp	.L30
.L29:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L30
.L27:
	movl	rear(%rip), %eax
	addl	$1, %eax
	movl	%eax, rear(%rip)
	movl	front(%rip), %eax
	addl	$1, %eax
	movl	%eax, front(%rip)
	movq	$10, -16(%rbp)
	jmp	.L30
.L39:
	nop
.L30:
	jmp	.L36
.L40:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L38
	call	__stack_chk_fail@PLT
.L38:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	insert, .-insert
	.section	.rodata
.LC6:
	.string	"Underflow"
	.text
	.globl	delete
	.type	delete, @function
delete:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$11, -8(%rbp)
.L72:
	cmpq	$21, -8(%rbp)
	ja	.L73
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L44(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L44(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L44:
	.long	.L59-.L44
	.long	.L58-.L44
	.long	.L57-.L44
	.long	.L73-.L44
	.long	.L56-.L44
	.long	.L73-.L44
	.long	.L55-.L44
	.long	.L54-.L44
	.long	.L53-.L44
	.long	.L73-.L44
	.long	.L52-.L44
	.long	.L51-.L44
	.long	.L50-.L44
	.long	.L49-.L44
	.long	.L48-.L44
	.long	.L73-.L44
	.long	.L47-.L44
	.long	.L73-.L44
	.long	.L46-.L44
	.long	.L73-.L44
	.long	.L45-.L44
	.long	.L74-.L44
	.text
.L46:
	movl	front(%rip), %eax
	movl	%eax, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L60
.L56:
	movl	front(%rip), %edx
	movl	rear(%rip), %eax
	cmpl	%eax, %edx
	jne	.L61
	movq	$10, -8(%rbp)
	jmp	.L60
.L61:
	movq	$18, -8(%rbp)
	jmp	.L60
.L48:
	movl	-16(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L60
.L50:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L63
	movq	$13, -8(%rbp)
	jmp	.L60
.L63:
	movq	$4, -8(%rbp)
	jmp	.L60
.L53:
	movl	rear(%rip), %eax
	cmpl	%eax, -20(%rbp)
	jg	.L65
	movq	$20, -8(%rbp)
	jmp	.L60
.L65:
	movq	$7, -8(%rbp)
	jmp	.L60
.L58:
	movl	rear(%rip), %eax
	cmpl	%eax, -20(%rbp)
	jg	.L67
	movq	$0, -8(%rbp)
	jmp	.L60
.L67:
	movq	$14, -8(%rbp)
	jmp	.L60
.L47:
	addl	$1, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L60
.L51:
	movq	$6, -8(%rbp)
	jmp	.L60
.L49:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$21, -8(%rbp)
	jmp	.L60
.L55:
	movl	$0, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L60
.L52:
	movl	$-1, front(%rip)
	movl	$-1, rear(%rip)
	movq	$21, -8(%rbp)
	jmp	.L60
.L59:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	4+pjob(%rip), %rax
	movl	(%rdx,%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L70
	movq	$2, -8(%rbp)
	jmp	.L60
.L70:
	movq	$16, -8(%rbp)
	jmp	.L60
.L54:
	movl	rear(%rip), %eax
	subl	$1, %eax
	movl	%eax, rear(%rip)
	movq	$21, -8(%rbp)
	jmp	.L60
.L57:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	4+pjob(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L60
.L45:
	movl	-20(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	pjob(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	leaq	pjob(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movl	-20(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	4+pjob(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	leaq	4+pjob(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	addl	$1, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L60
.L73:
	nop
.L60:
	jmp	.L72
.L74:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	delete, .-delete
	.section	.rodata
.LC7:
	.string	"\nInvalid choice:"
	.align 8
.LC8:
	.string	"\n1.Insert\t 2.Display\t 3.Delete\t 4.Exit"
.LC9:
	.string	"Enter your choice: "
.LC10:
	.string	"%d"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, pjob(%rip)
	movl	$0, 4+pjob(%rip)
	movl	$0, 8+pjob(%rip)
	movl	$0, 12+pjob(%rip)
	movl	$0, 16+pjob(%rip)
	movl	$0, 20+pjob(%rip)
	movl	$0, 24+pjob(%rip)
	movl	$0, 28+pjob(%rip)
	movl	$0, 32+pjob(%rip)
	movl	$0, 36+pjob(%rip)
	nop
.L76:
	movl	$-1, rear(%rip)
	nop
.L77:
	movl	$-1, front(%rip)
	nop
.L78:
	movq	$0, _TIG_IZ_93cQ_envp(%rip)
	nop
.L79:
	movq	$0, _TIG_IZ_93cQ_argv(%rip)
	nop
.L80:
	movl	$0, _TIG_IZ_93cQ_argc(%rip)
	nop
	nop
.L81:
.L82:
#APP
# 133 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-93cQ--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_93cQ_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_93cQ_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_93cQ_envp(%rip)
	nop
	movq	$13, -16(%rbp)
.L100:
	cmpq	$13, -16(%rbp)
	ja	.L102
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L85(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L85(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L85:
	.long	.L92-.L85
	.long	.L102-.L85
	.long	.L102-.L85
	.long	.L91-.L85
	.long	.L90-.L85
	.long	.L89-.L85
	.long	.L102-.L85
	.long	.L88-.L85
	.long	.L102-.L85
	.long	.L102-.L85
	.long	.L87-.L85
	.long	.L86-.L85
	.long	.L102-.L85
	.long	.L84-.L85
	.text
.L90:
	call	display
	movq	$10, -16(%rbp)
	jmp	.L93
.L91:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -16(%rbp)
	jmp	.L93
.L86:
	call	delete
	movq	$10, -16(%rbp)
	jmp	.L93
.L84:
	movq	$10, -16(%rbp)
	jmp	.L93
.L89:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L94
	cmpl	$4, %eax
	jg	.L95
	cmpl	$3, %eax
	je	.L96
	cmpl	$3, %eax
	jg	.L95
	cmpl	$1, %eax
	je	.L97
	cmpl	$2, %eax
	je	.L98
	jmp	.L95
.L94:
	movq	$0, -16(%rbp)
	jmp	.L99
.L96:
	movq	$11, -16(%rbp)
	jmp	.L99
.L98:
	movq	$4, -16(%rbp)
	jmp	.L99
.L97:
	movq	$7, -16(%rbp)
	jmp	.L99
.L95:
	movq	$3, -16(%rbp)
	nop
.L99:
	jmp	.L93
.L87:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$5, -16(%rbp)
	jmp	.L93
.L92:
	movl	$0, %edi
	call	exit@PLT
.L88:
	call	insert
	movq	$10, -16(%rbp)
	jmp	.L93
.L102:
	nop
.L93:
	jmp	.L100
	.cfi_endproc
.LFE4:
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
