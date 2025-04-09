	.file	"ahmetfurkaneq_DataStructure_1_flatten.c"
	.text
	.globl	front
	.bss
	.align 8
	.type	front, @object
	.size	front, 8
front:
	.zero	8
	.globl	_TIG_IZ_XP5I_argv
	.align 8
	.type	_TIG_IZ_XP5I_argv, @object
	.size	_TIG_IZ_XP5I_argv, 8
_TIG_IZ_XP5I_argv:
	.zero	8
	.globl	_TIG_IZ_XP5I_envp
	.align 8
	.type	_TIG_IZ_XP5I_envp, @object
	.size	_TIG_IZ_XP5I_envp, 8
_TIG_IZ_XP5I_envp:
	.zero	8
	.globl	rear
	.align 8
	.type	rear, @object
	.size	rear, 8
rear:
	.zero	8
	.globl	_TIG_IZ_XP5I_argc
	.align 4
	.type	_TIG_IZ_XP5I_argc, @object
	.size	_TIG_IZ_XP5I_argc, 4
_TIG_IZ_XP5I_argc:
	.zero	4
	.globl	temp
	.align 8
	.type	temp, @object
	.size	temp, 8
temp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d \n"
	.align 8
.LC1:
	.string	"Kuyrukta yazd\304\261r\304\261lacak eleman yok..."
.LC2:
	.string	"cls"
	.text
	.globl	printqueue
	.type	printqueue, @function
printqueue:
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
.L17:
	cmpq	$9, -8(%rbp)
	ja	.L18
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
	.long	.L18-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L18-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L8:
	movq	temp(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	temp(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, temp(%rip)
	movq	$7, -8(%rbp)
	jmp	.L11
.L5:
	movq	front(%rip), %rax
	movq	%rax, temp(%rip)
	movq	$7, -8(%rbp)
	jmp	.L11
.L9:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L11
.L3:
	movq	front(%rip), %rax
	testq	%rax, %rax
	jne	.L13
	movq	$3, -8(%rbp)
	jmp	.L11
.L13:
	movq	$8, -8(%rbp)
	jmp	.L11
.L7:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	system@PLT
	movq	$9, -8(%rbp)
	jmp	.L11
.L6:
	movq	temp(%rip), %rax
	testq	%rax, %rax
	je	.L15
	movq	$4, -8(%rbp)
	jmp	.L11
.L15:
	movq	$1, -8(%rbp)
	jmp	.L11
.L18:
	nop
.L11:
	jmp	.L17
.L19:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	printqueue, .-printqueue
	.section	.rodata
.LC3:
	.string	"\n data : %d"
.LC4:
	.string	"\n adres : %p"
	.text
	.globl	tekyazdir
	.type	tekyazdir, @function
tekyazdir:
.LFB2:
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
.L26:
	cmpq	$2, -8(%rbp)
	je	.L21
	cmpq	$2, -8(%rbp)
	ja	.L27
	cmpq	$0, -8(%rbp)
	je	.L28
	cmpq	$1, -8(%rbp)
	jne	.L27
	movq	$2, -8(%rbp)
	jmp	.L24
.L21:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L24
.L27:
	nop
.L24:
	jmp	.L26
.L28:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	tekyazdir, .-tekyazdir
	.section	.rodata
.LC5:
	.string	"Gecersiz secim."
	.align 8
.LC6:
	.string	"Kuyruga eleman eklemek icin Enqueue :1 "
	.align 8
.LC7:
	.string	"Kuyruktan eleman Cikarmak icin Dequeue :2 "
.LC8:
	.string	"Ekrana yazdirmak icin :3 "
.LC9:
	.string	"Secim yapiniz :"
.LC10:
	.string	"%d"
.LC11:
	.string	"Eklenecek Sayiyi giriniz:"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, rear(%rip)
	nop
.L30:
	movq	$0, front(%rip)
	nop
.L31:
	movq	$0, temp(%rip)
	nop
.L32:
	movq	$0, _TIG_IZ_XP5I_envp(%rip)
	nop
.L33:
	movq	$0, _TIG_IZ_XP5I_argv(%rip)
	nop
.L34:
	movl	$0, _TIG_IZ_XP5I_argc(%rip)
	nop
	nop
.L35:
.L36:
#APP
# 155 "ahmetfurkaneq_DataStructure_1.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XP5I--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_XP5I_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_XP5I_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_XP5I_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L52:
	cmpq	$13, -16(%rbp)
	ja	.L54
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L39(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L39(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L39:
	.long	.L45-.L39
	.long	.L44-.L39
	.long	.L54-.L39
	.long	.L43-.L39
	.long	.L54-.L39
	.long	.L42-.L39
	.long	.L41-.L39
	.long	.L54-.L39
	.long	.L54-.L39
	.long	.L54-.L39
	.long	.L40-.L39
	.long	.L54-.L39
	.long	.L54-.L39
	.long	.L38-.L39
	.text
.L44:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -16(%rbp)
	jmp	.L46
.L43:
	movq	$5, -16(%rbp)
	jmp	.L46
.L38:
	call	dequeue
	movq	$5, -16(%rbp)
	jmp	.L46
.L41:
	call	printqueue
	movq	$5, -16(%rbp)
	jmp	.L46
.L42:
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
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L46
.L40:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	enqueue
	movq	$5, -16(%rbp)
	jmp	.L46
.L45:
	movl	-24(%rbp), %eax
	cmpl	$3, %eax
	je	.L47
	cmpl	$3, %eax
	jg	.L48
	cmpl	$1, %eax
	je	.L49
	cmpl	$2, %eax
	je	.L50
	jmp	.L48
.L47:
	movq	$6, -16(%rbp)
	jmp	.L51
.L50:
	movq	$13, -16(%rbp)
	jmp	.L51
.L49:
	movq	$10, -16(%rbp)
	jmp	.L51
.L48:
	movq	$1, -16(%rbp)
	nop
.L51:
	jmp	.L46
.L54:
	nop
.L46:
	jmp	.L52
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	enqueue
	.type	enqueue, @function
enqueue:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$6, -16(%rbp)
.L68:
	cmpq	$7, -16(%rbp)
	ja	.L69
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L58(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L58(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L58:
	.long	.L63-.L58
	.long	.L69-.L58
	.long	.L70-.L58
	.long	.L61-.L58
	.long	.L60-.L58
	.long	.L69-.L58
	.long	.L59-.L58
	.long	.L57-.L58
	.text
.L60:
	movq	rear(%rip), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	rear(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	-24(%rbp), %rax
	movq	%rax, rear(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	tekyazdir
	movq	$2, -16(%rbp)
	jmp	.L64
.L61:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$7, -16(%rbp)
	jmp	.L64
.L59:
	movq	$3, -16(%rbp)
	jmp	.L64
.L63:
	movq	-24(%rbp), %rax
	movq	%rax, front(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, rear(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	tekyazdir
	movq	$2, -16(%rbp)
	jmp	.L64
.L57:
	movq	front(%rip), %rax
	testq	%rax, %rax
	jne	.L65
	movq	$0, -16(%rbp)
	jmp	.L64
.L65:
	movq	$4, -16(%rbp)
	jmp	.L64
.L69:
	nop
.L64:
	jmp	.L68
.L70:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	enqueue, .-enqueue
	.section	.rodata
.LC12:
	.string	"Kuyruk Bos.."
	.text
	.globl	dequeue
	.type	dequeue, @function
dequeue:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L89:
	cmpq	$10, -8(%rbp)
	ja	.L90
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L74(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L74(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L74:
	.long	.L90-.L74
	.long	.L82-.L74
	.long	.L81-.L74
	.long	.L80-.L74
	.long	.L79-.L74
	.long	.L78-.L74
	.long	.L77-.L74
	.long	.L76-.L74
	.long	.L90-.L74
	.long	.L75-.L74
	.long	.L73-.L74
	.text
.L79:
	movq	temp(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$6, -8(%rbp)
	jmp	.L83
.L82:
	movq	front(%rip), %rax
	movq	$0, 16(%rax)
	movq	$4, -8(%rbp)
	jmp	.L83
.L80:
	movq	front(%rip), %rax
	testq	%rax, %rax
	jne	.L84
	movq	$10, -8(%rbp)
	jmp	.L83
.L84:
	movq	$9, -8(%rbp)
	jmp	.L83
.L75:
	movq	front(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	front(%rip), %rax
	movq	%rax, temp(%rip)
	movq	front(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, front(%rip)
	movq	$2, -8(%rbp)
	jmp	.L83
.L77:
	movl	-12(%rbp), %eax
	jmp	.L86
.L78:
	movq	$0, rear(%rip)
	movq	$4, -8(%rbp)
	jmp	.L83
.L73:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L83
.L76:
	movl	$-1, %eax
	jmp	.L86
.L81:
	movq	front(%rip), %rax
	testq	%rax, %rax
	je	.L87
	movq	$1, -8(%rbp)
	jmp	.L83
.L87:
	movq	$5, -8(%rbp)
	jmp	.L83
.L90:
	nop
.L83:
	jmp	.L89
.L86:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	dequeue, .-dequeue
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
