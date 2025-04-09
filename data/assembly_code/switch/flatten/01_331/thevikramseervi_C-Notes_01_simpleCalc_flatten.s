	.file	"thevikramseervi_C-Notes_01_simpleCalc_flatten.c"
	.text
	.globl	_TIG_IZ_xOts_argc
	.bss
	.align 4
	.type	_TIG_IZ_xOts_argc, @object
	.size	_TIG_IZ_xOts_argc, 4
_TIG_IZ_xOts_argc:
	.zero	4
	.globl	_TIG_IZ_xOts_envp
	.align 8
	.type	_TIG_IZ_xOts_envp, @object
	.size	_TIG_IZ_xOts_envp, 8
_TIG_IZ_xOts_envp:
	.zero	8
	.globl	_TIG_IZ_xOts_argv
	.align 8
	.type	_TIG_IZ_xOts_argv, @object
	.size	_TIG_IZ_xOts_argv, 8
_TIG_IZ_xOts_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"enter the expression : "
.LC1:
	.string	"%d %c %d"
	.align 8
.LC2:
	.string	"division by zero is not possible"
.LC3:
	.string	"%d %c %d = %d\n"
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
	movq	$0, _TIG_IZ_xOts_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_xOts_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_xOts_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-xOts--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_xOts_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_xOts_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_xOts_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L33:
	cmpq	$17, -16(%rbp)
	ja	.L36
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L36-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L36-.L8
	.long	.L11-.L8
	.long	.L36-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rcx
	leaq	-29(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$17, -16(%rbp)
	jmp	.L21
.L10:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L19:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	imull	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L21
.L18:
	movq	$4, -16(%rbp)
	jmp	.L21
.L9:
	movq	$7, -16(%rbp)
	jmp	.L21
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L11:
	movl	-28(%rbp), %eax
	movl	-24(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L21
.L7:
	movzbl	-29(%rbp), %eax
	movsbl	%al, %eax
	subl	$37, %eax
	cmpl	$10, %eax
	ja	.L23
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L29-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L23-.L25
	.long	.L26-.L25
	.long	.L23-.L25
	.long	.L24-.L25
	.text
.L24:
	movq	$0, -16(%rbp)
	jmp	.L30
.L29:
	movq	$6, -16(%rbp)
	jmp	.L30
.L28:
	movq	$1, -16(%rbp)
	jmp	.L30
.L26:
	movq	$13, -16(%rbp)
	jmp	.L30
.L27:
	movq	$10, -16(%rbp)
	jmp	.L30
.L23:
	movq	$16, -16(%rbp)
	nop
.L30:
	jmp	.L21
.L15:
	movl	-28(%rbp), %eax
	movl	-24(%rbp), %ecx
	cltd
	idivl	%ecx
	movl	%edx, -20(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L21
.L16:
	movl	-28(%rbp), %eax
	movl	-24(%rbp), %esi
	cltd
	idivl	%esi
	movl	%eax, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L21
.L13:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L21
.L20:
	movl	-24(%rbp), %eax
	testl	%eax, %eax
	je	.L31
	movq	$5, -16(%rbp)
	jmp	.L21
.L31:
	movq	$15, -16(%rbp)
	jmp	.L21
.L14:
	movl	-24(%rbp), %ecx
	movzbl	-29(%rbp), %eax
	movsbl	%al, %edx
	movl	-28(%rbp), %eax
	movl	-20(%rbp), %esi
	movl	%esi, %r8d
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L21
.L36:
	nop
.L21:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
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
