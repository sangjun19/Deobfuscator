	.file	"jayeshbhandarkar_Data-Structures-and-Algorithms_q_flatten.c"
	.text
	.globl	queue_arr
	.bss
	.align 16
	.type	queue_arr, @object
	.size	queue_arr, 20
queue_arr:
	.zero	20
	.globl	front
	.align 4
	.type	front, @object
	.size	front, 4
front:
	.zero	4
	.globl	_TIG_IZ_Krce_argc
	.align 4
	.type	_TIG_IZ_Krce_argc, @object
	.size	_TIG_IZ_Krce_argc, 4
_TIG_IZ_Krce_argc:
	.zero	4
	.globl	_TIG_IZ_Krce_envp
	.align 8
	.type	_TIG_IZ_Krce_envp, @object
	.size	_TIG_IZ_Krce_envp, 8
_TIG_IZ_Krce_envp:
	.zero	8
	.globl	rear
	.align 4
	.type	rear, @object
	.size	rear, 4
rear:
	.zero	4
	.globl	_TIG_IZ_Krce_argv
	.align 8
	.type	_TIG_IZ_Krce_argv, @object
	.size	_TIG_IZ_Krce_argv, 8
_TIG_IZ_Krce_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Queue underflow "
	.text
	.globl	del
	.type	del, @function
del:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$6, -8(%rbp)
.L13:
	cmpq	$6, -8(%rbp)
	ja	.L15
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
	.long	.L8-.L4
	.long	.L15-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L15-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L6:
	cmpl	$0, -12(%rbp)
	je	.L9
	movq	$5, -8(%rbp)
	jmp	.L11
.L9:
	movq	$0, -8(%rbp)
	jmp	.L11
.L3:
	call	isEmpty
	movl	%eax, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L11
.L5:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L8:
	movl	front(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue_arr(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -16(%rbp)
	movl	front(%rip), %eax
	addl	$1, %eax
	movl	%eax, front(%rip)
	movq	$2, -8(%rbp)
	jmp	.L11
.L7:
	movl	-16(%rbp), %eax
	jmp	.L14
.L15:
	nop
.L11:
	jmp	.L13
.L14:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	del, .-del
	.section	.rodata
.LC1:
	.string	"Queue is empty"
.LC2:
	.string	"Queue is:"
.LC3:
	.string	"%d"
.LC4:
	.string	"\n"
	.text
	.globl	display
	.type	display, @function
display:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$11, -8(%rbp)
.L33:
	cmpq	$11, -8(%rbp)
	ja	.L34
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L19(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L19(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L19:
	.long	.L35-.L19
	.long	.L34-.L19
	.long	.L25-.L19
	.long	.L34-.L19
	.long	.L24-.L19
	.long	.L35-.L19
	.long	.L22-.L19
	.long	.L21-.L19
	.long	.L34-.L19
	.long	.L34-.L19
	.long	.L20-.L19
	.long	.L18-.L19
	.text
.L24:
	cmpl	$0, -12(%rbp)
	je	.L27
	movq	$6, -8(%rbp)
	jmp	.L29
.L27:
	movq	$10, -8(%rbp)
	jmp	.L29
.L18:
	call	isEmpty
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L29
.L22:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L29
.L20:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	front(%rip), %eax
	movl	%eax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L29
.L21:
	movl	rear(%rip), %eax
	cmpl	%eax, -16(%rbp)
	jg	.L31
	movq	$2, -8(%rbp)
	jmp	.L29
.L31:
	movq	$5, -8(%rbp)
	jmp	.L29
.L25:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue_arr(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L29
.L34:
	nop
.L29:
	jmp	.L33
.L35:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	display, .-display
	.section	.rodata
.LC5:
	.string	"\n 1.Insert"
.LC6:
	.string	"\n 2.Delete"
.LC7:
	.string	"\n 3.Display all element"
.LC8:
	.string	"\n 4.Quit"
.LC9:
	.string	"\n Enter your choice:   "
	.align 8
.LC10:
	.string	"Input the element for adding in queue:     "
.LC11:
	.string	"wrong choice "
.LC12:
	.string	"Deleted elements is %d"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$-1, front(%rip)
	nop
.L37:
	movl	$-1, rear(%rip)
	nop
.L38:
	movl	$0, -20(%rbp)
	jmp	.L39
.L40:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue_arr(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L39:
	cmpl	$4, -20(%rbp)
	jle	.L40
	nop
.L41:
	movq	$0, _TIG_IZ_Krce_envp(%rip)
	nop
.L42:
	movq	$0, _TIG_IZ_Krce_argv(%rip)
	nop
.L43:
	movl	$0, _TIG_IZ_Krce_argc(%rip)
	nop
	nop
.L44:
.L45:
#APP
# 170 "jayeshbhandarkar_Data-Structures-and-Algorithms_q.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Krce--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Krce_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Krce_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Krce_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L63:
	cmpq	$14, -16(%rbp)
	ja	.L65
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L48(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L48(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L48:
	.long	.L65-.L48
	.long	.L55-.L48
	.long	.L54-.L48
	.long	.L65-.L48
	.long	.L65-.L48
	.long	.L65-.L48
	.long	.L53-.L48
	.long	.L65-.L48
	.long	.L52-.L48
	.long	.L51-.L48
	.long	.L65-.L48
	.long	.L50-.L48
	.long	.L49-.L48
	.long	.L65-.L48
	.long	.L47-.L48
	.text
.L47:
	call	display
	movq	$1, -16(%rbp)
	jmp	.L56
.L49:
	movl	-28(%rbp), %eax
	cmpl	$4, %eax
	je	.L57
	cmpl	$4, %eax
	jg	.L58
	cmpl	$3, %eax
	je	.L59
	cmpl	$3, %eax
	jg	.L58
	cmpl	$1, %eax
	je	.L60
	cmpl	$2, %eax
	je	.L61
	jmp	.L58
.L57:
	movq	$8, -16(%rbp)
	jmp	.L62
.L59:
	movq	$14, -16(%rbp)
	jmp	.L62
.L61:
	movq	$6, -16(%rbp)
	jmp	.L62
.L60:
	movq	$11, -16(%rbp)
	jmp	.L62
.L58:
	movq	$9, -16(%rbp)
	nop
.L62:
	jmp	.L56
.L52:
	movl	$1, %edi
	call	exit@PLT
.L55:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -16(%rbp)
	jmp	.L56
.L50:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	insert
	movq	$1, -16(%rbp)
	jmp	.L56
.L51:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L56
.L53:
	call	del
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L56
.L54:
	movq	$1, -16(%rbp)
	jmp	.L56
.L65:
	nop
.L56:
	jmp	.L63
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	isEmpty
	.type	isEmpty, @function
isEmpty:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$2, -8(%rbp)
.L80:
	cmpq	$5, -8(%rbp)
	ja	.L81
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L69(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L69(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L69:
	.long	.L81-.L69
	.long	.L73-.L69
	.long	.L72-.L69
	.long	.L71-.L69
	.long	.L70-.L69
	.long	.L68-.L69
	.text
.L70:
	movl	$0, %eax
	jmp	.L74
.L73:
	movl	$1, %eax
	jmp	.L74
.L71:
	movl	$1, %eax
	jmp	.L74
.L68:
	movl	rear(%rip), %eax
	leal	1(%rax), %edx
	movl	front(%rip), %eax
	cmpl	%eax, %edx
	jne	.L75
	movq	$1, -8(%rbp)
	jmp	.L77
.L75:
	movq	$4, -8(%rbp)
	jmp	.L77
.L72:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L78
	movq	$3, -8(%rbp)
	jmp	.L77
.L78:
	movq	$5, -8(%rbp)
	jmp	.L77
.L81:
	nop
.L77:
	jmp	.L80
.L74:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	isEmpty, .-isEmpty
	.section	.rodata
.LC13:
	.string	"Queue overflow "
	.text
	.globl	insert
	.type	insert, @function
insert:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$7, -8(%rbp)
.L99:
	cmpq	$8, -8(%rbp)
	ja	.L100
	movq	-8(%rbp), %rax
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
	.long	.L100-.L85
	.long	.L92-.L85
	.long	.L91-.L85
	.long	.L90-.L85
	.long	.L89-.L85
	.long	.L101-.L85
	.long	.L87-.L85
	.long	.L86-.L85
	.long	.L101-.L85
	.text
.L89:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L93
	movq	$2, -8(%rbp)
	jmp	.L95
.L93:
	movq	$3, -8(%rbp)
	jmp	.L95
.L92:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L95
.L90:
	movl	rear(%rip), %eax
	addl	$1, %eax
	movl	%eax, rear(%rip)
	movl	rear(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	queue_arr(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$8, -8(%rbp)
	jmp	.L95
.L87:
	cmpl	$0, -12(%rbp)
	je	.L97
	movq	$1, -8(%rbp)
	jmp	.L95
.L97:
	movq	$4, -8(%rbp)
	jmp	.L95
.L86:
	call	isFull
	movl	%eax, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L95
.L91:
	movl	$0, front(%rip)
	movq	$3, -8(%rbp)
	jmp	.L95
.L100:
	nop
.L95:
	jmp	.L99
.L101:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	insert, .-insert
	.globl	isFull
	.type	isFull, @function
isFull:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$2, -8(%rbp)
.L111:
	cmpq	$3, -8(%rbp)
	je	.L103
	cmpq	$3, -8(%rbp)
	ja	.L112
	cmpq	$0, -8(%rbp)
	je	.L105
	cmpq	$2, -8(%rbp)
	je	.L106
	jmp	.L112
.L103:
	movl	$0, %eax
	jmp	.L107
.L105:
	movl	$1, %eax
	jmp	.L107
.L106:
	movl	rear(%rip), %eax
	cmpl	$4, %eax
	jne	.L108
	movq	$0, -8(%rbp)
	jmp	.L110
.L108:
	movq	$3, -8(%rbp)
	jmp	.L110
.L112:
	nop
.L110:
	jmp	.L111
.L107:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	isFull, .-isFull
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
