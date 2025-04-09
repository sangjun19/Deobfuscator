	.file	"ajipal_first-year_practice_flatten.c"
	.text
	.globl	_TIG_IZ_JRgv_argc
	.bss
	.align 4
	.type	_TIG_IZ_JRgv_argc, @object
	.size	_TIG_IZ_JRgv_argc, 4
_TIG_IZ_JRgv_argc:
	.zero	4
	.globl	_TIG_IZ_JRgv_argv
	.align 8
	.type	_TIG_IZ_JRgv_argv, @object
	.size	_TIG_IZ_JRgv_argv, 8
_TIG_IZ_JRgv_argv:
	.zero	8
	.globl	_TIG_IZ_JRgv_envp
	.align 8
	.type	_TIG_IZ_JRgv_envp, @object
	.size	_TIG_IZ_JRgv_envp, 8
_TIG_IZ_JRgv_envp:
	.zero	8
	.text
	.globl	room_size
	.type	room_size, @function
room_size:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L4:
	cmpq	$0, -8(%rbp)
	jne	.L7
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	jmp	.L6
.L7:
	nop
	jmp	.L4
.L6:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	room_size, .-room_size
	.globl	item_cost
	.type	item_cost, @function
item_cost:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$3, -8(%rbp)
.L20:
	cmpq	$7, -8(%rbp)
	ja	.L22
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L11(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L11(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L11:
	.long	.L22-.L11
	.long	.L22-.L11
	.long	.L15-.L11
	.long	.L14-.L11
	.long	.L22-.L11
	.long	.L13-.L11
	.long	.L12-.L11
	.long	.L10-.L11
	.text
.L14:
	movq	$6, -8(%rbp)
	jmp	.L16
.L12:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L16
.L13:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-16(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -16(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L16
.L10:
	movss	-16(%rbp), %xmm0
	jmp	.L21
.L15:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L18
	movq	$5, -8(%rbp)
	jmp	.L16
.L18:
	movq	$7, -8(%rbp)
	jmp	.L16
.L22:
	nop
.L16:
	jmp	.L20
.L21:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	item_cost, .-item_cost
	.section	.rodata
.LC1:
	.string	"House Renovation"
.LC2:
	.string	"How many room/s: "
.LC3:
	.string	"%d"
.LC4:
	.string	"length: "
.LC5:
	.string	"width: "
	.align 8
.LC6:
	.string	"Enter number of items per room: "
	.align 8
.LC7:
	.string	"====================================="
	.align 8
.LC8:
	.string	"Total of House Renovation: %.2f"
.LC9:
	.string	"item %d: "
.LC10:
	.string	"%s"
.LC11:
	.string	"cost: "
.LC12:
	.string	"%f"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$2304, %rsp
	movl	%edi, -10468(%rbp)
	movq	%rsi, -10480(%rbp)
	movq	%rdx, -10488(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_JRgv_envp(%rip)
	nop
.L24:
	movq	$0, _TIG_IZ_JRgv_argv(%rip)
	nop
.L25:
	movl	$0, _TIG_IZ_JRgv_argc(%rip)
	nop
	nop
.L26:
.L27:
#APP
# 140 "ajipal_first-year_practice.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-JRgv--0
# 0 "" 2
#NO_APP
	movl	-10468(%rbp), %eax
	movl	%eax, _TIG_IZ_JRgv_argc(%rip)
	movq	-10480(%rbp), %rax
	movq	%rax, _TIG_IZ_JRgv_argv(%rip)
	movq	-10488(%rbp), %rax
	movq	%rax, _TIG_IZ_JRgv_envp(%rip)
	nop
	movq	$3, -10424(%rbp)
.L40:
	cmpq	$9, -10424(%rbp)
	ja	.L43
	movq	-10424(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L30(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L30(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L30:
	.long	.L35-.L30
	.long	.L43-.L30
	.long	.L43-.L30
	.long	.L34-.L30
	.long	.L43-.L30
	.long	.L33-.L30
	.long	.L43-.L30
	.long	.L32-.L30
	.long	.L31-.L30
	.long	.L29-.L30
	.text
.L31:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-10452(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-10444(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-10448(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-10448(%rbp), %edx
	movl	-10444(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	room_size
	movl	%eax, -10432(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-10440(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -10436(%rbp)
	movq	$9, -10424(%rbp)
	jmp	.L36
.L34:
	movq	$8, -10424(%rbp)
	jmp	.L36
.L29:
	movl	-10440(%rbp), %eax
	cmpl	%eax, -10436(%rbp)
	jge	.L37
	movq	$0, -10424(%rbp)
	jmp	.L36
.L37:
	movq	$5, -10424(%rbp)
	jmp	.L36
.L33:
	movl	-10440(%rbp), %edx
	leaq	-10416(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	item_cost
	movd	%xmm0, %eax
	movl	%eax, -10428(%rbp)
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	pxor	%xmm1, %xmm1
	cvtss2sd	-10428(%rbp), %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$7, -10424(%rbp)
	jmp	.L36
.L35:
	movl	-10436(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-10416(%rbp), %rcx
	movl	-10436(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	addq	$4, %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-10416(%rbp), %rcx
	movl	-10436(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -10436(%rbp)
	movq	$9, -10424(%rbp)
	jmp	.L36
.L32:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L41
	jmp	.L42
.L43:
	nop
.L36:
	jmp	.L40
.L42:
	call	__stack_chk_fail@PLT
.L41:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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
